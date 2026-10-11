#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#define assertm(exp, msg) assert((void(msg), exp))

using namespace std;

template<typename T, size_t Width, size_t Height>
class Frame {

public:
    Frame(size_t stride = Width) : m_img(Width * Height), m_stride(stride) {}
    ~Frame() = default;

    const T& operator()(size_t row, size_t col) const {
        assertm(row < Height and col < Width, "indices are bounded by dimension?");
        return m_img[row * m_stride + col];
    }
    
    T& operator()(size_t row, size_t col) {
        assertm(row < Height and col < Width, "indices are bounded by dimension?");
        return m_img[row * m_stride + col];
    }

    T* pixels() { return m_img.data(); }
    const T* pixels() const { return m_img.data(); }

    size_t stride() const { return m_stride; }

private:
    vector<T> m_img;
    size_t m_stride;
};

enum class SlotState : uint8_t {
    FREE,
    READY,
    OCCUPIED
};

struct SlotReclaimer {
    std::atomic<SlotState>* state;
    void operator()(Frame<int, 8, 6>*) const { state->store(SlotState::FREE); }
};

using FrameHandle = std::unique_ptr<Frame<int, 8, 6>, SlotReclaimer>;

struct ProdConsumerShared {
    std::mutex m_mutex;
    condition_variable m_cv;
    atomic<size_t> customerSubscribedSlot = numeric_limits<size_t>::max();
    atomic<SlotState> slotView {SlotState::FREE};
    FrameHandle customerFrameView {nullptr, SlotReclaimer{&slotView}};    
};

void worker(ProdConsumerShared &state) {
    using ImageFrame = Frame<int, 8, 6>;
    state.customerSubscribedSlot.store(1);
    constexpr size_t MAX_CUSTOMER_FRAME = 10;
    size_t framesProcessed = 0;
    while (framesProcessed < MAX_CUSTOMER_FRAME) {
        FrameHandle handle {nullptr, SlotReclaimer{&state.slotView}};
        {
            unique_lock<mutex> lk(state.m_mutex);
            state.m_cv.wait(lk, [&state]{ return state.customerFrameView != nullptr;});
            handle = std::move(state.customerFrameView);
            state.slotView.store(SlotState::OCCUPIED);
        }
        cout << "Holding frame:" << framesProcessed << endl;
        this_thread::sleep_for(chrono::milliseconds(10));
        framesProcessed++;
    }

};

constexpr size_t RING_SIZE {4};
struct ProducerBuffers {
    using ImageFrame = Frame<int, 8, 6>;
    array<ImageFrame, RING_SIZE> buffers;
    array<SlotState, RING_SIZE> slotStates;
};

int main()
{
    Frame<int, 6, 4> frame;
    auto pixels = frame.pixels();
    pixels[3] = 10;
    cout << "Image pixel at (0,3)=" << frame(0,3) << std::endl;
    cout << "Before: Image pixel at (1,2)=" << frame(1,2) << std::endl;
    pixels[8] = 100;
    cout << "After: Image pixel at (1,2)=" << frame(1,2) << std::endl;


    constexpr size_t FRAMES_TO_PROCESS {20};
    using ImageFrame = Frame<int, 8, 6>;
    ProducerBuffers prodBuffers {};
    size_t frameCounter = 0;

    ProdConsumerShared state;
    thread consumer(worker, std::ref(state));
    while (frameCounter < FRAMES_TO_PROCESS) {
        size_t slot = frameCounter % RING_SIZE;
        this_thread::sleep_for(chrono::milliseconds(33));
        frameCounter++;
        if (slot != state.customerSubscribedSlot.load()) {
            prodBuffers.buffers[slot] = ImageFrame();
            prodBuffers.slotStates[slot] = SlotState::READY;
            continue;
        } else {
            if (prodBuffers.slotStates[slot] != SlotState::OCCUPIED) {
                lock_guard<mutex> lk(state.m_mutex);
                prodBuffers.buffers[slot] = ImageFrame();
                prodBuffers.slotStates[slot] = SlotState::READY;
                state.slotView = prodBuffers.slotStates[slot];
                state.customerFrameView = FrameHandle(&prodBuffers.buffers[slot], SlotReclaimer{&state.slotView});
                state.m_cv.notify_one();
                cout << "Frame Ready!" << "Buffer Slot=" << slot << endl;
            } else {
                cout << "New Frame available but slot still occupied!" << endl;
                continue;
            }
        }
    }
    consumer.join();
}