#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <vector>
#include <stack>

using namespace std;
template<typename T>
using point = pair<T, T>;

enum class DIRECTION {
    NONE,
    LEFT,
    RIGHT,
    UP,
    DOWN,
};

// A parent vertex can have two (Left/Right) childern
// Leaf vertex would have none
template<typename T, size_t D = 2>
struct Vertex {
    std::unique_ptr<Vertex> left;
    std::unique_ptr<Vertex> right;
    point<T> vertex; 
};

template<typename T, size_t D = 2>
class Tree {

public:
    explicit Tree(const vector<point<T>>& inputPoints, optional<point<T>> val = nullopt) : m_depth(0)
    {
        m_root = make_unique<Vertex<T>>();
        if (not val.has_value()) {
            // find median of inputs make the new root accordingly
            vector<point<T>> sortedPoints (inputPoints);
            sort(sortedPoints.begin(), sortedPoints.end());
            size_t medianIndex = sortedPoints.size() / 2;
            m_root->vertex = sortedPoints[medianIndex];
        } else {
            m_root->left = nullptr;
            m_root->right = nullptr;
            m_root->vertex = val.value();
        }
        // Build the tree
        for (auto& point : inputPoints) {
            insert(point);
        }
        m_stack.push(m_root.get());
    }
    void show() {
        std::cout << "Printing the contents of tree with depth=" << m_depth << std::endl;
        list(m_root.get());
    }

    point<T> nearest(const point<T>& p) {

        size_t minDistance = std::numeric_limits<size_t>::max();
        size_t level = 0;
        DIRECTION dir = DIRECTION::DOWN;
        while (not m_stack.empty()) {
            auto currentVertex = m_stack.top();
            if (currentVertex->left == nullptr && currentVertex->right == nullptr) {
                size_t distance = (currentVertex->vertex.first - p.first) * (currentVertex->vertex.first - p.first) +
                            (currentVertex->vertex.second - p.second) * (currentVertex->vertex.second - p.second);
                m_stack.pop();
                minDistance = distance;
                std::cout << "Reached leaf node with distance: " << distance << std::endl;
               dir = DIRECTION::UP;
                continue;
            }
            if (dir == DIRECTION::DOWN) {
                level = level % D;
                if (level == 0) {
                    if (p.first < currentVertex->vertex.first) {
                        if (currentVertex->left != nullptr) {
                            m_stack.push(currentVertex->left.get());
                        }
                    } else {
                        if (currentVertex->right != nullptr) {
                            m_stack.push(currentVertex->right.get());
                        }
                    }
                } else if (p.second < currentVertex->vertex.second) {
                    if (currentVertex->left != nullptr) {
                        m_stack.push(currentVertex->left.get());
                    }
                } else {
                    if (currentVertex->right != nullptr) {
                        m_stack.push(currentVertex->right.get());
                    }
                }
                std::cout << "Going down the tree with Distance= " << minDistance << std::endl;
                level++;
            } else {
                size_t distance = (currentVertex->vertex.first - p.first) * (currentVertex->vertex.first - p.first) +
                            (currentVertex->vertex.second - p.second) * (currentVertex->vertex.second - p.second);
                if (distance <= minDistance) {
                    minDistance = distance;
                    std::cout << "Reached minimum Distance= " << minDistance << std::endl;
                    m_nearestPoint = currentVertex->vertex;
                }
                m_stack.pop();
                std::cout << "Going up the tree with Distance= " << minDistance << std::endl;
            }
        }
        cout << "Finished processing the tree withDistance= " << minDistance << std::endl;
        return m_nearestPoint;
    }
    
    ~Tree() = default;
private:
    unique_ptr<Vertex<T>> m_root;
    size_t m_depth;
    stack<Vertex<T>*> m_stack;
    point<T> m_nearestPoint;
    void insert(const point<T>& p) {
        size_t level = 0;
        DIRECTION dir = DIRECTION::NONE;
        Vertex<T>* currentVertex = m_root.get();
        while (level <= m_depth) {
            std::cout << "Processing vertex: (" << currentVertex->vertex.first << ", " << currentVertex->vertex.second << ")" << std::endl;
            if (level % D == 0) {
                dir = p.first < currentVertex->vertex.first ? DIRECTION::LEFT : DIRECTION::RIGHT;
            } else {
                dir = p.second < currentVertex->vertex.second ? DIRECTION::LEFT : DIRECTION::RIGHT;
            }
            if (dir == DIRECTION::LEFT) {
                if (currentVertex-> left == nullptr) {
                    std::cout << "Inserting left vertex: (" << p.first << ", " << p.second << ")" << std::endl;
                    currentVertex->left = make_unique<Vertex<T>>();
                    currentVertex->left->vertex = p;
                    if (currentVertex->right == nullptr) { 
                        std::cout << "No right neighbor! Increasing depth to " << (m_depth + 1) << std::endl;
                        m_depth++; 
                    } 
                    break;
                } else {
                    currentVertex = currentVertex->left.get();
                }
            } else if (dir == DIRECTION::RIGHT) {
                if (currentVertex->right == nullptr) {
                    std::cout << "Inserting right vertex: (" << p.first << ", " << p.second << ")" << std::endl;
                    currentVertex->right = make_unique<Vertex<T>>();
                    currentVertex->right->vertex = p;
                    if (currentVertex->left == nullptr) { 
                        std::cout << "No left neighbor! Increasing depth to " << (m_depth + 1) << std::endl;
                        m_depth++; 
                    };
                    break;
                } else {
                    currentVertex = currentVertex->right.get();
                }
            }
            level++;
        }
    }
    void list(Vertex<T>* vertex) {
        if (vertex == nullptr) return;
        if (vertex->left == nullptr and vertex->right == nullptr) {
            std::cout << "Leaf: (" << vertex->vertex.first << ", " << vertex->vertex.second << ")" << std::endl;
        } else {
            std::cout << "Parent: (" << vertex->vertex.first << ", " << vertex->vertex.second << ")" << std::endl;
        }
        if (vertex->left != nullptr) {
            std::cout << "--->Left Child : (" << vertex->left->vertex.first << ", " << vertex->left->vertex.second << ")" << std::endl;
        }
        if (vertex->right != nullptr) {
            std::cout << "--->Right Child : (" << vertex->right->vertex.first << ", " << vertex->right->vertex.second << ")" << std::endl;
        }
        list(vertex->left.get());
        list(vertex->right.get());
    }
};

int main()
{
    Tree<int> tree({point<int>(2,7), point<int>(17,15), point<int>(9,1), point<int>(10,19), point<int>(13,15), point<int>(6,12)}, point<int>(3,6));
    tree.show();

    //Tree<int> tree2({point<int>(2,2), point<int>(4,7), point<int>(1,3), point<int>(2,4), point<int>(5,4), point<int>(7,2)}, point<int>(3,6));
    //tree2.show();

    auto nearest = tree.nearest(point<int>(7,2));
    std::cout << "Nearest point in the tree: (" << nearest.first << ", " << nearest.second << ")" << std::endl;
}