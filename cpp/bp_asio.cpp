#include <boost/asio.hpp>
#include <boost/process.hpp>

#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>

namespace bp = boost::process;
namespace bas = boost::asio;

// Yet another Async Command Executor class
// based on boost process and ASIO
// Good short description of boost::asio async event loop
// https://stackoverflow.com/questions/15568100/confused-when-boostasioio-service-run-method-blocks-unblocks
class YACE
{

    std::string m_command; // command to be executed in a new process
    std::vector<std::string> m_args; // arguments for command
    bp::child m_child; // Child process running command
    bas::io_service m_IoProcessor; // ASIO service running event loop for I/O processing
    // Error/output channels between parent and child process
    std::vector<char> m_outBuf {std::vector<char>(128)};
    bp::async_pipe m_outPipe {bp::async_pipe(m_IoProcessor)};

public:
    explicit YACE(std::string inC, std::vector<std::string>(inA)):
        m_command(inC), m_args(inA)
    {
        std::cout << "YACE created" << std::endl;
    }

    // Launch command with group
    void launch()
    {
        m_child = bp::child(m_command, m_args, bp::std_in.close(), (bp::std_out & bp::std_err) > m_outPipe,
             m_IoProcessor);
        if (m_child.valid()) {
            std::cout << "Child process successfully created" << std::endl;
        } else {
            std::cout << "Child failed, exception must be thrown" << std::endl;
        }
    }

    // process command collect logs
    std::future<std::string> process()
    {
        auto fAsync = std::async(std::launch::async, [this]()->std::string {
            std::string outVal = "";
            // Mixing sync and async operations is not a good idea
            // but here we really need to keep running until
            // the process running command is in action
            while(this->m_child.running()) {
                bas::async_read(m_outPipe, bas::buffer(m_outBuf), [this, &outVal](const boost::system::error_code &ec, std::size_t size)
                {
                    std::cout << "Output log from " << m_command << " received with size=" << size << " and error=" << ec.value() << std::endl;
                    if (ec.value() != 0) {  outVal = ec.message(); return; }
                    std::string rcvdString(m_outBuf.data(), size);
                    std::cout << rcvdString << std::flush << std::endl;
                });
                // This will run the ASIO event loop until a handler exist (above) and has data to consume
                // in case no data is available it will block current thread of execution
                try {
                    m_IoProcessor.run();
                    std::cout << "Resetting IO processor for " << m_command << std::endl;
                    // To keep running the event loop again, in case all data was consumed by handler
                    m_IoProcessor.reset();
                } catch (std::exception& ex) {
                    std::cout << "Re-running " << m_command << " failed with error=" << ex.what() << std::endl;
                    break;
                }
            }
            return outVal;
        });

        return fAsync;
    }
    void terminate() {
        if (m_child.running()) {
            std::cout << "Terminating running" << m_command << "signal=" << std::endl;
            m_child.terminate();
        } else {
            std::cout << "command=" << m_command << " already dead" << std::endl;
        }
    }
};

int main(void)
{
    std::vector<std::string> arg1;
    arg1.push_back("localhost");

    YACE c1("/bin/ping", arg1);

    std::vector<std::string> arg2;
    arg2.push_back("-L");
    YACE c2("/bin/netstat", arg2);

    c1.launch();
    c2.launch();
    auto f1 = c1.process();
    auto f2 = c2.process();

    try {
        std::cout << "netstat wait returned:" << f2.get();
    } catch (std::exception& ex) {
        std::cout << "Getting data from netstat failed" << ex.what() << std::endl;
        c2.terminate();
    }
    try {
        std::cout << "Ping wait returned:" << f1.get();
    } catch (std::exception& ex) {
        std::cout << "Getting data from ping failed with" << ex.what() << std::endl;
        c1.terminate();
    }
    std::cout << "Program ending, should not reach in normal cases" << std::endl;

    return 0;
}
