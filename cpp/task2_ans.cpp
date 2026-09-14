#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

/* In machine learning, we sometimes have to find the nearest neighbour for a
 * given point. This can be efficiently done using k-d trees. In this task,
 * you will implement a nearest-neighbour search in euclidean space using a
 * k-d tree.
 *
 * Compile with g++ -std=c++17 task2.cpp -o task2 */

constexpr int pointDims = 3;
typedef std::array<float, pointDims> Point;

void printPoint(Point &point);
struct Node {
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
    Point point;

    Node(Point &pt) : left(nullptr), right(nullptr), point(pt) { }
};
enum class DIRECTION {
    LEFT,
    RIGHT
};

class KDTree {
public:
    KDTree(std::vector<Point> &points) {
        root_ = constructTree(points, 0);
    }

    Point nearestNeighbour(Point queryPoint) const {
        auto [node, dist] = nearestPointInSubtree(queryPoint, root_, 0);
        return node->point;
    }

private:
    std::shared_ptr<Node> constructTree(std::vector<Point> &points, int depth) {
        if (points.size() == 0) {
            return nullptr;
        }

        int axis = depth % pointDims;
        std::sort(points.begin(), points.end(),
                  [&] (const Point &a, const Point &b) {
                      return a[axis] < b[axis];
                });
        int medianIdx = points.size() / 2;
        auto node = std::make_shared<Node>(points[medianIdx]);
        std::vector<Point> leftPts(medianIdx);
        std::copy(points.begin(), points.begin() + medianIdx, leftPts.begin());
        std::vector<Point> rightPts(points.size() - medianIdx - 1);
        std::copy(points.begin() + medianIdx + 1, points.end(), rightPts.begin());
        node->left = constructTree(leftPts, depth + 1);
        node->right = constructTree(rightPts, depth + 1);
        return node;
    }

    std::tuple<std::shared_ptr<Node>, float> nearestPointInSubtree(
        Point queryPoint, std::shared_ptr<Node> rootNode, int depth) const {
        // We can reach here if a subtree has only one child node/vertex
        if (!rootNode) {
            std::cerr << "Not a valid node in KDTree" << std::endl;
            return {nullptr, std::numeric_limits<float>::max()};
        }

        int axis = depth % pointDims;
        float axisDistToRoot = queryPoint[axis] - rootNode->point[axis];
        float distToRoot = 0.0;
        for (int i = 0; i < pointDims; ++i) {
            distToRoot += std::pow(queryPoint[i] - rootNode->point[i], 2.0);
        }
        distToRoot = std::pow(distToRoot, 0.5);

        if (!rootNode->left && !rootNode->right) {
            std::cout << "Leaf node reached with Distance=" << distToRoot << std::endl;
            return {rootNode, distToRoot};
        }

        /**********************************************************************
         * TODO: Implement the remaining part of this function by calling it
         * recursively. Also take care of the edge cases (e.g. one child node).
         * This can be done as follows:
         * 1: Figure out on which side of the separating hyperplane the query
         *    point lies and recursively descend into that part of the tree.
         * 2: If the distance of the query point to the best point in that
         *    sub-tree is smaller than the distance to the separating hyperplane
         *    (axisDistToRoot), just return the best point and corresponding
         *    distance from that sub-tree.
         * 3: Otherwise the closest point to the query point can also be the
         *    root node or a point in the other sub-tree. Calculate all
         *    distances and return the node with the smallest distance to the
         *    query point.
         **********************************************************************/

        // 1. Hyperplane selection is based on the level of tree we are currently processing
        // For a 3D point, this could be 0,1,2, on each level we need to decide
        // downwards direction
        if (axis >= pointDims) {
            std::cerr << "Unexpected Axis level, must be {0,1,2}" << std::endl;
            return {rootNode, distToRoot};
        }
        std::shared_ptr<Node> selectedNode = (queryPoint[axis] >= rootNode->point[axis]) ? rootNode->right : rootNode->left;
        std::shared_ptr<Node> adjacentNode = (queryPoint[axis] >= rootNode->point[axis]) ? rootNode->left  : rootNode->right;

        std::cout << "Starting recursion at depth=" << depth << " with axis=" << axis << std::endl;
        auto [nearestNode, minDistance] = nearestPointInSubtree(queryPoint, selectedNode, depth + 1);
        // 2. Update nearest point in the subtree based on comparison with current root Node
        if (distToRoot < minDistance) {
            nearestNode = rootNode;
            minDistance = distToRoot;
            printPoint(rootNode->point);
            std::cout << "New minimum point found in selected plane with Distance=" << minDistance << std::endl;
        }
        // 3. Evaluate distance from hyperplane
        float planeDistance = std::abs(axisDistToRoot);
        // Check also adjacent branch in current sub-tree in case distance from hyperplane is smaller
        if (planeDistance < minDistance) {
            auto [altNode, altDistance] = nearestPointInSubtree(queryPoint, adjacentNode, depth + 1);
            if (altDistance < minDistance) {
                nearestNode = altNode;
                minDistance = altDistance;
                printPoint(nearestNode->point);
                std::cout << "New minimum point found in alternate plane with Distance=" << minDistance << std::endl;
            }
        }
        
        return {nearestNode, minDistance};
    }

    std::shared_ptr<Node> root_;
};

void printPoint(Point &point) {
    std::cout << "(";
    for (int i = 0; i < pointDims; ++i) {
        std::cout << point[i];
        if (i + 1 < pointDims) {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl << std::endl;
};

int main() {
    // Insert some points to the tree
    std::vector<Point> points {
        {0.935, 0.086, 0.069},
        {0.407, 0.5  , 0.349},
        {0.959, 0.394, 0.004},
        {0.418, 0.608, 0.452},
        {0.331, 0.704, 0.418},
        {0.76 , 0.988, 0.544},
        {0.89 , 0.063, 0.137},
        {0.574, 0.903, 0.101},
        {0.9  , 0.889, 0.708},
        {0.322, 0.963, 0.816}
    };
    KDTree tree(points);

    std::cout << "Tree successfully constructed!" << std::endl;
    // Find nearest neighbour to a given point
    std::cout << "Starting first test...." << std::endl;
    Point nn = tree.nearestNeighbour({0.5, 0.5, 0.5});
    std::cout << "I found the following nearest neighbour for (0.5, 0.5, 0.5),"
              << std::endl << "[expected (0.418, 0.608, 0.452)]:" << std::endl;
    printPoint(nn);

    std::cout << "Starting second test...." << std::endl;
    nn = tree.nearestNeighbour({0.2, 0.7, 0.8});
    std::cout << "I found the following nearest neighbour for (0.2, 0.7, 0.8),"
              << std::endl << " [expected (0.322, 0.963, 0.816)]:" << std::endl;
    printPoint(nn);

    std::cout << "Starting third test...." << std::endl;
    nn = tree.nearestNeighbour({0.7, 0.2, 0.5});
    std::cout << "I found the following nearest neighbour for (0.7, 0.2, 0.5),"
              << std::endl << " [expected (0.89 , 0.063, 0.137)]:" << std::endl;
    printPoint(nn);

    return 0;
}
