#pragma once
#include <vector>
#include <amp.h>
#include <amp_graphics.h>
#include <amp_math.h>
#include <amp_short_vectors.h>
#include <ppl.h>


namespace hpc
{
    static __declspec(thread) int tls_slotIdx;
    
    const int slotNumber = 3;
    const int UnitThreadsNumber = MAXIMUM_WAIT_OBJECTS;

    class GroupWaitThreadProc
    {
    public:
        GroupWaitThreadProc(HANDLE signal, HANDLE* pendings, int size)
            : m_signal(signal)
            , m_pendingSignals(pendings)
            , m_size(size)
        {}

        GroupWaitThreadProc(const GroupWaitThreadProc& source)
            : m_signal(source.m_signal)
            , m_pendingSignals(source.m_pendingSignals)
            , m_size(source.m_size)
        {}


        void operator()()
        {
            DWORD ret = WaitForMultipleObjects(m_size, m_pendingSignals, TRUE, INFINITE);
            assert(ret >= WAIT_OBJECT_0 && ret < (WAIT_OBJECT_0 + m_size));

            SetEvent(m_signal);
        }

    protected:
        HANDLE m_signal;
        HANDLE* m_pendingSignals;
        int m_size;
    };

    class tile_barrier_cpu
    {
    public:
        tile_barrier_cpu(int hitBound)
            : m_hitBound(hitBound)
        {
            m_groupBound = (hitBound - 1) / UnitThreadsNumber + 1;

            for (int i = 0; i < slotNumber; i++)
            {
                m_events[i] = m_groupEvents[i] = nullptr;
            }
        }

        tile_barrier_cpu()
            : m_hitBound(0)
        {
            m_groupBound = 0;

            for (int i = 0; i < slotNumber; i++)
            {
                m_events[i] = m_groupEvents[i] = nullptr;
            }
        }

        void bindEvents(HANDLE** events, HANDLE** groupEvents)
        {
            for (int i = 0; i < slotNumber; i++)
            {
                m_events[i] = events[i];
                m_groupEvents[i] = groupEvents[i];
            }
        }

        void tls_init(unsigned int index)
        {
            assert(m_events && m_groupEvents);

            tls_slotIdx = 0;
        }

        void setHitBound(int bound) 
        {
            m_hitBound = bound;
            m_groupBound = (bound - 1) / UnitThreadsNumber + 1;
        }

        void wait(unsigned int index)
        {
            assert(m_events && m_groupEvents);

            std::thread groupThread;

            if (index % UnitThreadsNumber == 0)
            {
                int groupId = index / UnitThreadsNumber;
                int size = ((groupId + 1) * UnitThreadsNumber <= m_hitBound ? UnitThreadsNumber : (m_hitBound % UnitThreadsNumber));
                GroupWaitThreadProc threadProc(m_groupEvents[tls_slotIdx][groupId]
                    , m_events[tls_slotIdx] + groupId * UnitThreadsNumber
                    , size);

                groupThread.swap(std::thread(threadProc));
            }

            SetEvent(m_events[tls_slotIdx][index]);
            DWORD ret = WaitForMultipleObjects(m_groupBound, m_groupEvents[tls_slotIdx], TRUE, INFINITE);
            assert((ret >= WAIT_OBJECT_0) && (ret < (WAIT_OBJECT_0 + m_groupBound)));

            int resetSlot = (((tls_slotIdx - 1) % slotNumber) + slotNumber) % slotNumber;
            tls_slotIdx = ((tls_slotIdx + 1) % slotNumber);

            ResetEvent(m_events[resetSlot][index]);
            if (groupThread.joinable())
            {
                groupThread.join();
                ResetEvent(m_groupEvents[resetSlot][index / UnitThreadsNumber]);
            }
        }


    protected:
        int m_hitBound;
        int m_groupBound;
        HANDLE* m_events[slotNumber];
        HANDLE* m_groupEvents[slotNumber];

        volatile long m_testCount = 0;
    };

    class ConcurrencyThreadsNumber
    {
    public:
        ConcurrencyThreadsNumber(int number)
        {
            m_defaultPolicy = Concurrency::CurrentScheduler::GetPolicy();
            Concurrency::CurrentScheduler::Create((Concurrency::SchedulerPolicy(2,
                Concurrency::MinConcurrency, number,
                Concurrency::MaxConcurrency, number)));
        }

        ~ConcurrencyThreadsNumber()
        {
            Concurrency::CurrentScheduler::Create(m_defaultPolicy);
        }
    protected:
        Concurrency::SchedulerPolicy m_defaultPolicy;
    };


    template <int Dim0, int Dim1 = 0, int Dim2 = 0>
    class cpu_tiled_index : public concurrency::tiled_index<Dim0, Dim1, Dim2>
    {
    public:
        cpu_tiled_index(const concurrency::index<3>& _Global,
            const concurrency::index<3>& _Local,
            const concurrency::index<3>& _Tile,
            const concurrency::index<3>& _Tile_origin,
            const concurrency::tile_barrier& _Barrier,
            tile_barrier_cpu& barrier_cpu) __CPU_ONLY
            : concurrency::tiled_index<Dim0, Dim1, Dim2>(_Global, _Local, _Tile, _Tile_origin, _Barrier)
            , m_cpu_barrier(barrier_cpu)
        {
            m_localIndex = local[0] * (Dim1 * Dim2) + local[1] * Dim2 + local[2];
        }

        cpu_tiled_index(const cpu_tiled_index& other) __CPU_ONLY
            : concurrency::tiled_index<Dim0, Dim1, Dim2>(other)
            , m_cpu_barrier(other.m_cpu_barrier)
        {
            m_localIndex = local[0] * (Dim1 * Dim2) + local[1] * Dim2 + local[2];
        }

        tile_barrier_cpu& barrier_cpu() __CPU_ONLY
        {
            return m_cpu_barrier;
        }

        void wait() __CPU_ONLY
        {
            m_cpu_barrier.wait(m_localIndex);
        }

        unsigned int index() const __CPU_ONLY
        {
            return m_localIndex;
        }

    protected:
        tile_barrier_cpu& m_cpu_barrier;
        unsigned int m_localIndex;
    };

    template <int Dim0, int Dim1>
    class cpu_tiled_index<Dim0, Dim1, 0> : public concurrency::tiled_index<Dim0, Dim1>
    {
    public:
        cpu_tiled_index(const concurrency::index<2>& _Global,
            const concurrency::index<2>& _Local,
            const concurrency::index<2>& _Tile,
            const concurrency::index<2>& _Tile_origin,
            const concurrency::tile_barrier& _Barrier,
            tile_barrier_cpu& barrier_cpu) __CPU_ONLY
            : concurrency::tiled_index<Dim0, Dim1>(_Global, _Local, _Tile, _Tile_origin, _Barrier)
            , m_cpu_barrier(barrier_cpu)
        {
            m_localIndex = local[0] * Dim1 + local[1];
        }

        cpu_tiled_index(const cpu_tiled_index& other) __CPU_ONLY
            : concurrency::tiled_index<Dim0, Dim1>(other)
            , m_cpu_barrier(other.m_cpu_barrier)
        {
            m_localIndex = local[0] * Dim1 + local[1];
        }

        tile_barrier_cpu& barrier_cpu() __CPU_ONLY
        {
            return m_cpu_barrier;
        }

        void wait() __CPU_ONLY
        {
            m_cpu_barrier.wait(m_localIndex);
        }

        unsigned int index() const __CPU_ONLY
        {
            return m_localIndex;
        }

    protected:
        tile_barrier_cpu& m_cpu_barrier;
        unsigned int m_localIndex;
    };

    template <int Dim0>
    class cpu_tiled_index<Dim0, 0, 0> : public concurrency::tiled_index<Dim0>
    {
    public:
        cpu_tiled_index(const concurrency::index<1>& _Global,
            const concurrency::index<1>& _Local,
            const concurrency::index<1>& _Tile,
            const concurrency::index<1>& _Tile_origin,
            const concurrency::tile_barrier& _Barrier,
            tile_barrier_cpu& barrier_cpu) __CPU_ONLY
            : concurrency::tiled_index<Dim0>(_Global, _Local, _Tile, _Tile_origin, _Barrier)
            , m_cpu_barrier(barrier_cpu)
        {
            m_localIndex = local[0];
        }

        cpu_tiled_index(const cpu_tiled_index& other) __CPU_ONLY
            : concurrency::tiled_index<Dim0>(other)
            , m_cpu_barrier(other.m_cpu_barrier)
        {
            m_localIndex = local[0];
        }

        tile_barrier_cpu& barrier_cpu() __CPU_ONLY
        {
            return m_cpu_barrier;
        }

        void wait() __CPU_ONLY
        {
            m_cpu_barrier.wait(m_localIndex);
        }

        unsigned int index() const __CPU_ONLY
        {
            return m_localIndex;
        }

    protected:
        tile_barrier_cpu& m_cpu_barrier;
        unsigned int m_localIndex;
    };

    class dummy_tiled_barrier
    {
    public:
        concurrency::tile_barrier* get()
        {
            return reinterpret_cast<concurrency::tile_barrier*>(m_mem);
        }

        concurrency::tile_barrier& get_ref()
        {
            return *get();
        }

    protected:
        BYTE m_mem[sizeof(concurrency::tile_barrier)];
    };

    template <typename tile_index_type, typename kernelType>
    class cpu_thread_main
    {
    public:
        cpu_thread_main(tile_index_type& index
                      , const kernelType& kernel)
            : m_tile_index(index)
            , m_kernel(kernel)
        { }

        cpu_thread_main(const cpu_thread_main& source)
            : m_tile_index(source.m_tile_index)
            , m_kernel(source.m_kernel)
        {}

        void operator()()
        {
            m_tile_index.barrier_cpu().tls_init(m_tile_index.index());
            m_kernel(m_tile_index);
        }

    protected:
        mutable tile_index_type m_tile_index;
        const kernelType& m_kernel;
    };

    template <typename kernelType, typename cpu_tiled_index_type, typename thread_entry_type>
    class PPLTask
    {
    public:
        PPLTask(const kernelType& kernel)
            : m_kernel(kernel)
        {

        }

        void operator()(cpu_tiled_index_type t_index) const
        {
            thread_entry_type entry(t_index, m_kernel);
            entry();
        }
    protected:
        const kernelType& m_kernel;
    };

    class TiledProcessorBase
    {
    public:
        void clear()
        {
            clearEvents();
        }

        void createEvents(int total)
        {
            clearEvents();
            m_events.resize(slotNumber);
            std::vector<HANDLE*> headers;
            headers.resize(slotNumber);

            std::vector<HANDLE*> groupHeaders;
            groupHeaders.resize(slotNumber);

            for (size_t g = 0 ; g < m_events.size(); g++)
            {
                std::vector<HANDLE>& group = m_events[g];
                group.resize(total);

                for (int i = 0; i < total; i++)
                {
                    group[i] = CreateEvent(nullptr, TRUE, FALSE, NULL);
                    assert(group[i]);
                }

                headers[g] = group.data();
            }

            m_groupEvents.resize(slotNumber);
            int groupSize = (total - 1) / UnitThreadsNumber + 1;
            for (size_t g = 0; g < m_groupEvents.size(); g++)
            {
                std::vector<HANDLE>& group = m_groupEvents[g];
                group.resize(groupSize);

                for (size_t i = 0; i < groupSize; i++)
                {
                    group[i] = CreateEvent(nullptr, TRUE, FALSE, NULL);
                    assert(group[i]);
                }

                groupHeaders[g] = group.data();
            }

            m_cpu_barrier.bindEvents(headers.data(), groupHeaders.data());

        }

        void resetEvents()
        {
            resetEventsImp(m_events);
            resetEventsImp(m_groupEvents);
        }


        void clearEvents()
        {
            resetEvents();
            
            clearEventsImp(m_events);
            clearEventsImp(m_groupEvents);
        }
    protected:
        void resetEventsImp(std::vector<std::vector<HANDLE>>& events)
        {
            for (size_t i = 0; i < events.size(); i++)
            {
                std::vector<HANDLE>& group = events[i];
                for (auto handle : group)
                {
                    ResetEvent(handle);
                }
            }
        }
        void clearEventsImp(std::vector<std::vector<HANDLE>>& events)
        {
            for (size_t i = 0; i < events.size(); i++)
            {
                std::vector<HANDLE>& group = events[i];
                for (auto handle : group)
                {
                    CloseHandle(handle);
                }

                group.clear();
            }

            events.clear();
        }

    protected:
        std::vector<std::vector<HANDLE>> m_events;
        std::vector<std::vector<HANDLE>> m_groupEvents;
        tile_barrier_cpu m_cpu_barrier;
        dummy_tiled_barrier m_dummy_barrier;
    };

    template < typename kernelType , int d0, int d1 = 0, int d2 = 0>
    class TiledProcessor : public TiledProcessorBase
    {
    public:
        typedef concurrency::tiled_extent<d0, d1, d2> tiled_extent_type;
        typedef concurrency::tiled_index<d0, d1, d2> tiled_index_type;
        typedef cpu_tiled_index<d0, d1, d2> cpu_tiled_index_type;
        typedef typename cpu_thread_main<cpu_tiled_index_type, kernelType> thread_entry_type;
        typedef typename PPLTask<kernelType, cpu_tiled_index_type, thread_entry_type> task_type;

        TiledProcessor(const tiled_extent_type& domain, const kernelType& kernel)
            : m_domain(domain)
            , m_kernel(kernel)
        {}

        void run()
        {
            clear();

            int tileDimi = m_domain[0] / d0;
            int tileDimj = m_domain[1] / d1;
            int tileDimk = m_domain[2] / d2;

            int blockSize = d0 * d1 * d2;

            m_cpu_barrier.setHitBound(blockSize);
            createEvents(blockSize);

            ConcurrencyThreadsNumber PPLThreadNumber(blockSize + 1);

            for (int i = 0; i < tileDimi; i ++)
                for (int j = 0; j < tileDimj; j++)
                    for (int k = 0; k < tileDimk; k++)
                    {
                        concurrency::index<3> tile(i, j, k);
                        concurrency::index<3> tile_origin(i * d0, j * d1, k * d2);

                        resetEvents();

                        std::vector<cpu_tiled_index_type> tileArray;

                        for (int x = 0; x < d0; x++)
                            for (int y = 0; y < d1; y++)
                                for (int z = 0; z < d2; z++)
                                {
                                    concurrency::index<3> local(x, y, z);
                                    concurrency::index<3> global_index = tile_origin + local;

                                    cpu_tiled_index_type t_index(global_index, local, tile, tile_origin, m_dummy_barrier.get_ref(), m_cpu_barrier);
                                    tileArray.push_back(t_index);
                                }
                        task_type task(m_kernel);
                        concurrency::parallel_for_each(std::begin(tileArray), std::end(tileArray), task);
                    }

            clear();
        }

    protected:

        const kernelType& m_kernel;
        tiled_extent_type m_domain;
    };




    template <typename kernelType, int d0, int d1>
    class TiledProcessor<kernelType, d0, d1, 0> : public TiledProcessorBase
    {
    public:
        typedef concurrency::tiled_extent<d0, d1> tiled_extent_type;
        typedef concurrency::tiled_index<d0, d1> tiled_index_type;
        typedef cpu_tiled_index<d0, d1> cpu_tiled_index_type;
        typedef typename cpu_thread_main<cpu_tiled_index_type, kernelType> thread_entry_type;
        typedef typename PPLTask<kernelType, cpu_tiled_index_type, thread_entry_type> task_type;


        TiledProcessor(const tiled_extent_type& domain, const kernelType& kernel)
            : m_domain(domain)
            , m_kernel(kernel)
        {}
        
        void run()
        {
            clear();

            int tileDimi = m_domain[0] / d0;
            int tileDimj = m_domain[1] / d1;

            int blockSize = d0 * d1;
            m_cpu_barrier.setHitBound(blockSize);
            createEvents(blockSize);
            
            ConcurrencyThreadsNumber PPLThreadNumber(blockSize + 1);

            for (int i = 0; i < tileDimi; i++)
                for (int j = 0; j < tileDimj; j++)
                {
                    concurrency::index<2> tile(i, j);
                    concurrency::index<2> tile_origin(i * d0, j * d1);

                    resetEvents();
                    std::vector<cpu_tiled_index_type> tileArray;
                    for (int x = 0; x < d0; x++)
                        for (int y = 0; y < d1; y++)
                        {
                            concurrency::index<2> local(x, y);
                            concurrency::index<2> global_index = tile_origin + local;

                            cpu_tiled_index_type t_index(global_index, local, tile, tile_origin, m_dummy_barrier.get_ref(), m_cpu_barrier);

                            tileArray.push_back(t_index);
                        }

                    task_type task(m_kernel);
                    concurrency::parallel_for_each(std::begin(tileArray), std::end(tileArray), task);
                    //std::cout << i << "/" << tileDimi << " " << j << "/" << tileDimj << " Done." << std::endl;

                }

            clear();
        }
    protected:
        const kernelType& m_kernel;
        tiled_extent_type m_domain;
    };

    template <typename kernelType, int d0>
    class TiledProcessor<kernelType, d0, 0, 0> : public TiledProcessorBase
    {
    public:
        typedef concurrency::tiled_extent<d0> tiled_extent_type;
        typedef concurrency::tiled_index<d0> tiled_index_type;
        typedef cpu_tiled_index<d0> cpu_tiled_index_type;
        typedef typename cpu_thread_main<cpu_tiled_index_type, kernelType> thread_entry_type;
        typedef typename PPLTask<kernelType, cpu_tiled_index_type, thread_entry_type> task_type;

        TiledProcessor(const tiled_extent_type& domain, const kernelType& kernel)
            : m_domain(domain)
            , m_kernel(kernel)
        {}

        void run()
        {
            clear();
            int tileDim = m_domain[0] / d0;

            m_cpu_barrier.setHitBound(d0);
            createEvents(d0);

            ConcurrencyThreadsNumber PPLThreadNumber(d0 + 1);

            for (int i = 0; i < tileDim; i++)
            {
                concurrency::index<1> tile(i);
                concurrency::index<1> tile_origin(i * d0);

                resetEvents();
                std::vector<cpu_tiled_index_type> tileArray;
                for (int x = 0; x < d0; x++)
                {
                    concurrency::index<1> local(x);
                    concurrency::index<1> global_index = tile_origin + local;

                    cpu_tiled_index_type t_index(global_index, local, tile, tile_origin, m_dummy_barrier.get_ref(), m_cpu_barrier);
                    tileArray.push_back(t_index);
                }

                task_type task(m_kernel);
                concurrency::parallel_for_each(std::begin(tileArray), std::end(tileArray), task);
            }

            clear();
        }

    protected:
        const kernelType& m_kernel;
        tiled_extent_type m_domain;
    };

}