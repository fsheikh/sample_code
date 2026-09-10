#include <cassert>
#include <algorithm>
#include <iostream>
#include <memory>
#include <optional>
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

int main()
{
    Frame<int, 6, 4> frame;
    auto pixels = frame.pixels();
    pixels[3] = 10;
    cout << "Image pixel at (0,3)=" << frame(0,3) << std::endl;
    cout << "Before: Image pixel at (1,2)=" << frame(1,2) << std::endl;
    pixels[8] = 100;
    cout << "After: Image pixel at (1,2)=" << frame(1,2) << std::endl;
}