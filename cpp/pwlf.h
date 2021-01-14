/*
 * A generic piece-wise linear function
 *
 * Contributors:
 *       fahim.sheikh@gmail.com
 */

#pragma once

#include <array>
#include <cassert>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace pwlf {

class InfiniteSlope : public std::exception
{
};

class PointNotInRange : public std::exception
{
};
// A piecewise linear function composed of P-1 segments
// defined by P points, each point represented by a pair
// of x-y coordinates of type T.

// First an integral type to specialize for valid sizes
template<std::size_t P, std::size_t MIN_SIZE>
struct MinSegments : public std::integral_constant<bool, (P >= MIN_SIZE)>
{
};
template<typename T, std::size_t Points>
class PwLinFunc
{
    std::array<std::pair<T,T>, Points> m_points;
    std::array<T, Points - 1> m_slopes;

public:

    explicit PwLinFunc(std::array<std::pair<T,T>, Points> pointsA,
    typename std::enable_if<MinSegments<Points, 2>::value >::type* = nullptr)
    : m_points(pointsA)
    {

        assert(m_points.size() >= 2);

        // Special case one segment
        if (m_points.size() == 2) {
            if (m_points[1].first == m_points[0].first) {
                throw InfiniteSlope();
            } else {
                auto yDiff = m_points[1].second - m_points[0].second;
                auto xDiff = m_points[1].first - m_points[0].first;
                m_slopes[0] = yDiff / xDiff;

            }
        } else {
            auto pointsItr = m_points.begin() + 1;
            auto slopesItr = m_slopes.begin();
            while (pointsItr != m_points.end()) {
                auto prevPoint = pointsItr - 1;
                auto x1 = prevPoint->first;
                auto x2 = pointsItr->first;
                auto y1 = prevPoint->second;
                auto y2 = pointsItr->second;
                if (x1 == x2) {
                    throw InfiniteSlope();
                } else {
                    *slopesItr = (y2 - y1) / (x2 - x1);
                }
                slopesItr++;
                pointsItr++;
            }
        }
    }

    PwLinFunc(const PwLinFunc&) = delete;
    PwLinFunc(const PwLinFunc&&) = delete;

    // Produces ordinate (y-coordinate) given absicssa (x-coordinate)
    T operator()(const T& absicssa) {
        // TODO maintain a list of intervals one for each segment?
        // Special one-segment case
        if (m_points.size() == 2) {
            return m_slopes[0] * (absicssa - m_points[0].first) + m_points[0].second;
        }
        auto pointsItr = m_points.begin() + 1;
        std::size_t segmentIndex = 0;
        bool found = false;
        while (pointsItr != m_points.end()) {
            auto prevPoint = pointsItr - 1;
            if (absicssa >= prevPoint->first and absicssa < pointsItr->first) {
                found = true;
                break;
            }
            segmentIndex++;
            pointsItr++;
        }

        T y;
        if (not found) {
            throw PointNotInRange();
        } else {
            // Use point-slope form of straight line equation to get y-coordinate (ordinate)
            // against absicssa
            y = m_slopes[segmentIndex] * (absicssa - m_points[segmentIndex].first)
                + m_points[segmentIndex].second;
        }

        return y;
    }

private:
    std::size_t slope(const T& x);
    // No default piece-wise linear function possible for users
    PwLinFunc();
};

template<typename T, std::size_t S>
class ArrayOfPair {

    std::array<std::pair<T,T>, S> m_points;
public:
    explicit ArrayOfPair(std::array<std::pair<T,T>, S> input) : m_points(input)
    {}
};
} // namespace pwlf
