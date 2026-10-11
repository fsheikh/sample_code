#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <vector>

using namespace std;
constexpr size_t MAX_BOXES = 64;

struct Box {
    float x1, y1, x2, y2;   // top-left, bottom-right, absolute pixel coords
    float score;            // confidence, higher is better
};

static bool approx_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

static bool box_eq(const Box& a, const Box& b) {
    return approx_eq(a.x1, b.x1) && approx_eq(a.y1, b.y1)
        && approx_eq(a.x2, b.x2) && approx_eq(a.y2, b.y2)
        && approx_eq(a.score, b.score);
}

// Return IoU in [0, 1]. Assume boxes are valid (x2 > x1, y2 > y1).
float iou(const Box& a, const Box& b) {
    if (box_eq(a, b)) return 1.0f;
    float intersectionWidth, intersectionHeight;
    float intersectionArea;
    float unionArea;
    
    intersectionWidth = min(a.x2, b.x2) - max(a.x1, b.x1);
    intersectionHeight  = min(a.y2, b.y2) - max(a.y1, b.y1);
    if (intersectionWidth <= 0.0f or intersectionHeight <= 0.0f) {
        return 0.0f;
    }
    intersectionArea = intersectionWidth * intersectionHeight;
    float areaBoxA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaBoxB = (b.x2 - b.x1) * (b.y2 - b.y1);
    unionArea = areaBoxA + areaBoxB - intersectionArea;
    assert(unionArea > 0.0f);
    return intersectionArea / unionArea;
}

// In-place NMS. Sorts boxes by score (descending), then greedily keeps boxes
// whose IoU with any already-kept box is <= iou_threshold.
// Returns number of boxes written into 'out' (max max_out).
size_t nms(Box* boxes, size_t count, float iou_threshold, Box* out, size_t max_out) {
    if (count == 0) {
        return 0;
    }
    if (count == 1) {
        *out = boxes[0];
        return 1;
    }
    if (max_out < count) {
        cerr << "Warning! too low maximum output size=" << max_out << endl;
    }

    sort(boxes, boxes + count, [](const Box &a, const Box &b) { return a.score > b.score; });
    for (size_t idx = 0; idx < count; idx++) {
        cout << "Box(" << boxes[idx].x1 << "," << boxes[idx].y1 << "," << boxes[idx].x2
            << "," << boxes[idx].y2 << ")" << "Confidence Score=" << boxes[idx].score << endl;
    }

    size_t outCount = 0;
    for (size_t observeIdx = 0; observeIdx < count and outCount < max_out; observeIdx++) {
        size_t inCount = 0;
        while (inCount < outCount) {
            if (iou(boxes[observeIdx], out[inCount]) > iou_threshold) {
                break; // This box has an overlap with an alreadx existing box of high score
            }
            inCount++;
        }

        if (inCount == outCount) {
            // inner loop was completed with an overlap so box can be moved to output list
            out[outCount++] = boxes[observeIdx];
        }
    }
    return outCount;
}
static int g_failures = 0;

static void check(bool cond, const char* name) {
    if (cond) { printf("[PASS] %s\n", name); }
    else      { printf("[FAIL] %s\n", name); ++g_failures; }
}


static bool contains(const Box* out, size_t n, const Box& expected) {
    for (size_t i = 0; i < n; ++i)
        if (box_eq(out[i], expected)) return true;
    return false;
}

int main() {
    Box out[MAX_BOXES];

    // ---- Test 1: empty input ----
    {
        size_t n = nms(nullptr, 0, 0.5f, out, MAX_BOXES);
        check(n == 0, "empty input returns 0");
    }

    // ---- Test 2: single box ----
    {
        Box a = {0, 0, 10, 10, 0.9f};
        Box in[] = {a};
        size_t n = nms(in, 1, 0.5f, out, MAX_BOXES);
        check(n == 1 && box_eq(out[0], a), "single box kept");
    }

    // ---- Test 3: two non-overlapping boxes ----
    {
        Box a = {0,  0,  10, 10, 0.9f};
        Box b = {20, 20, 30, 30, 0.8f};
        Box in[] = {a, b};
        size_t n = nms(in, 2, 0.5f, out, MAX_BOXES);
        check(n == 2 && contains(out, n, a) && contains(out, n, b),
              "non-overlapping both kept");
    }
    // ---- Test 4: two overlapping, lower score suppressed ----
    {
        // IoU(a, b) ~= 0.68 > 0.5  -> b suppressed
        Box a = {0, 0, 10, 10, 0.9f};
        Box b = {1, 1, 11, 11, 0.8f};
        Box in[] = {a, b};
        size_t n = nms(in, 2, 0.5f, out, MAX_BOXES);
        check(n == 1 && box_eq(out[0], a),
              "overlapping lower score suppressed");
    }

    // ---- Test 5: identical boxes, highest score wins ----
    {
        Box lo = {0, 0, 10, 10, 0.7f};
        Box hi = {0, 0, 10, 10, 0.9f};
        Box in[] = {lo, hi};   // lower first, to test sorting
        size_t n = nms(in, 2, 0.5f, out, MAX_BOXES);
        check(n == 1 && box_eq(out[0], hi),
              "identical boxes keep highest score");
    }

    // ---- Test 6: three boxes, middle suppressed ----
    {
        Box a = {0,  0,  10, 10, 0.95f};
        Box b = {1,  1,  11, 11, 0.85f};   // overlaps a -> suppressed
        Box c = {50, 50, 60, 60, 0.75f};   // far -> kept
        Box in[] = {b, c, a};              // unsorted input
        size_t n = nms(in, 3, 0.5f, out, MAX_BOXES);
        check(n == 2 && contains(out, n, a) && contains(out, n, c),
              "three boxes, middle suppressed");
    }

    // ---- Test 7: output capacity respected ----
    {
        Box in[5] = {
            {0,   0,   5,   5,   0.9f},
            {10,  10,  15,  15,  0.8f},
            {20,  20,  25,  25,  0.7f},
            {30,  30,  35,  35,  0.6f},
            {40,  40,  45,  45,  0.5f},
        };
        size_t n = nms(in, 5, 0.5f, out, /*max_out=*/3);
        check(n == 3, "output capacity respected");
    }

    // ---- Test 8: IoU correctness on a known pair ----
    {
        Box a = {0, 0, 10, 10};
        Box b = {0, 0, 10, 10};   // identical
        Box c = {0, 0,  5,  5};   // a quarter of a
        Box d = {20, 20, 30, 30}; // disjoint
        check(approx_eq(iou(a, b), 1.0f), "IoU identical == 1");
        check(approx_eq(iou(a, c), 0.25f), "IoU quarter == 0.25");
        check(approx_eq(iou(a, d), 0.0f),  "IoU disjoint == 0");
    }
    printf("\n%s (%d failure%s)\n",
           g_failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
           g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}