#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

using namespace std;

constexpr size_t ARENA_SIZE {1024};
uint8_t PREALLOC_BUFFER[ARENA_SIZE];
using arenaslot = pair<uintptr_t, size_t>;

class CustomAllocator {
    list<arenaslot> m_freeSlots{arenaslot {reinterpret_cast<uintptr_t>(PREALLOC_BUFFER), ARENA_SIZE}};
    list<arenaslot> m_allocatedSlots;
    std::mutex m_lock;

    void merge(const arenaslot& freedSlot) {
        // Find insertion point: first slot with address > freedSlot.first
        auto nextItr = std::find_if(m_freeSlots.begin(), m_freeSlots.end(),
            [&](const arenaslot& s){ return s.first > freedSlot.first; });

        // Check previous slot
        bool mergePrev = false;
        if (nextItr != m_freeSlots.begin()) {
            auto prevItr = std::prev(nextItr);
            if (prevItr->first + prevItr->second == freedSlot.first) {
                prevItr->second += freedSlot.second;
                mergePrev = true;
            }
        }

        // Check next slot
        bool mergeNext = false;
        if (nextItr != m_freeSlots.end()) {
            if (freedSlot.first + freedSlot.second == nextItr->first) {
                if (mergePrev) {
                    // Merge all three: prev absorbs next
                    auto prevItr = std::prev(nextItr);
                    prevItr->second += nextItr->second;
                    m_freeSlots.erase(nextItr);
                } else {
                    // Merge with next only
                    nextItr->first = freedSlot.first;
                    nextItr->second += freedSlot.second;
                    mergeNext = true;
                }
            }
        }

        // If no merge happened, insert in sorted position
        if (!mergePrev && !mergeNext) {
            m_freeSlots.insert(nextItr, freedSlot);
        }
    }
public:
    CustomAllocator() = default;
    ~CustomAllocator() = default;

    uintptr_t allocate(size_t inSize) {
        uintptr_t slotAddress = 0;
        lock_guard<mutex> guard(m_lock);
        for (auto pos = m_freeSlots.begin(); pos != m_freeSlots.end(); pos++) {
            auto& currentSlot = *pos;
            if (inSize <= currentSlot.second ) {
                slotAddress = currentSlot.first;
                if (inSize == currentSlot.second) {
                    m_freeSlots.erase(pos);
                } else {
                    currentSlot.first += inSize;
                    currentSlot.second -= inSize;
                }
                m_allocatedSlots.push_back(arenaslot(slotAddress, inSize));
                return slotAddress;
            }
        }

        return 0;
    }

    void deallocate(uintptr_t ptr) {
        lock_guard<mutex> gaurd(m_lock);
        auto allocationFound = find_if(m_allocatedSlots.begin(), m_allocatedSlots.end(),
        [&ptr](const arenaslot& elem) {
           return elem.first == ptr;
        });
        if (allocationFound == m_allocatedSlots.end()) {
            ::cerr << "Allocation not found or double free" << endl;
            return;
        } else {
            ::cout << "Allocation found:" << hex << ptr << endl;
            auto freedSlot = *allocationFound;
            m_allocatedSlots.erase(allocationFound);
            merge(freedSlot);
        }
    }
};
void parallel_allocation(CustomAllocator& inAlloc) {

    cout << "Allocation from parallel thread" << endl;
    uintptr_t alloc64 = inAlloc.allocate(64);
    cout << "Parallel thread allocated:" << alloc64 << endl;
    this_thread::sleep_for(chrono::milliseconds(100));
    cout << "Deallocation from parallel thread" << endl;
    inAlloc.deallocate(alloc64);
}

int main()
{
    ::cout << "Start of buffer:" << ::hex<< reinterpret_cast<uintptr_t>(PREALLOC_BUFFER) << ::endl;
    ::cout << "End of buffer:" << ::hex << reinterpret_cast<uintptr_t>(PREALLOC_BUFFER + ARENA_SIZE) << ::endl;
    CustomAllocator cAlloc;
    thread t(parallel_allocation, ref(cAlloc));
    uintptr_t alloc16 = cAlloc.allocate(16);
    uintptr_t alloc2 = cAlloc.allocate(2);
    ::cout << "Allocation of 16:" << hex << alloc16 << ::endl;
    ::cout << "Allocation of 2:" << hex << alloc2 << ::endl;
    cAlloc.deallocate(alloc2);
    cAlloc.deallocate(alloc16);
    cAlloc.deallocate(alloc16);
    t.join(); 
}