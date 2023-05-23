#include "rvole.h"
#include <coproto/Socket/AsioSocket.h>
#include <libOTe/Tools/Coproto.h>
#include <cryptoTools/Network/Channel.h>
#include <cryptoTools/Common/Defines.h>
#include <macoro/task.h>
#include <macoro/macros.h>
using OtExtSender = osuCrypto::SilentVoleSender;
using OtExtReceiver = osuCrypto::SilentVoleReceiver;
using Socket = osuCrypto::Socket;
using AsioSocket = coproto::AsioSocket;
using PRNG = osuCrypto::PRNG;
using block = osuCrypto::block;
using Timer = osuCrypto::Timer;
using u8 = osuCrypto::u8;
using u64 = osuCrypto::u64;
using osuCrypto::cp::task;

template<typename T> using span = osuCrypto::span<T>;
enum Role {
    Sender = 1,
    Receiver = 2
};

std::unique_ptr<OtExtReceiver> receiver;
std::unique_ptr<OtExtSender> sender;
// std::unique_ptr<Timer> gtimer;
AsioSocket chl;
PRNG prng;

task<> sync(Socket& chl, Role role)
	{
		MC_BEGIN(task<>,&chl, role,
			dummy = u8{},
			timer = std::unique_ptr<Timer>{new Timer},
			start = Timer::timeUnit{},
			mid = Timer::timeUnit{},
			end = Timer::timeUnit{},
			ms = u64{},
			rrt = std::chrono::system_clock::duration{}
		);

		if (role == Role::Receiver)
		{

		 	MC_AWAIT(chl.recv(dummy));

			start = timer->setTimePoint("");

			MC_AWAIT(chl.send(dummy));
			MC_AWAIT(chl.recv(dummy));

			mid = timer->setTimePoint("");

			MC_AWAIT(chl.send(std::move(dummy)));

			rrt = mid - start;
			ms = std::chrono::duration_cast<std::chrono::milliseconds>(rrt).count();

			// wait for half the round trip time to start both parties at the same time.
			if (ms > 4)
				std::this_thread::sleep_for(rrt / 2);

		}
		else
		{
			MC_AWAIT(chl.send(dummy));
			MC_AWAIT(chl.recv(dummy));
			MC_AWAIT(chl.send(dummy));
			MC_AWAIT(chl.recv(dummy));
		}

		MC_END();
	}

extern "C" {

    void init_silent_vole(int role, const char* ip, const uint64_t seed[2], uint64_t vector_size) {
        std::string ip_str(ip);
        block s(seed[0], seed[1]);
        prng.SetSeed(s);
        // gtimer = std::make_unique<Timer>();
        if (role == Role::Sender) {
            if (sender.get() != NULL) {
                throw std::runtime_error("init should be called once" LOCATION);
            }
            sender = std::make_unique<OtExtSender>();
            chl = coproto::asioConnect(ip_str, true);

            sender->mMalType = osuCrypto::SilentSecType::SemiHonest;
            sender->mMultType = osuCrypto::MultType::slv5;
            // sender->mDebug = true;
            sender->configure(vector_size);
        } else if (role == Role::Receiver ) {
            if (receiver.get() != NULL) {
                throw std::runtime_error("init should be called once" LOCATION);
            }
            receiver = std::make_unique<OtExtReceiver>();
            chl = coproto::asioConnect(ip_str, false);
            receiver->mMalType = osuCrypto::SilentSecType::SemiHonest;
            receiver->mMultType = osuCrypto::MultType::slv5;
            // receiver->mDebug = true;
            receiver->configure(vector_size);
        } else {
            throw std::runtime_error("unkown role. " LOCATION);
        }
    }

    void silent_send_inplace(uint64_t delta[2], uint64_t num_ots) {
        if (sender.get() == NULL) {
            throw std::runtime_error("sender is null" LOCATION);
        }
        block d(delta[1], delta[0]);
        osuCrypto::cp::sync_wait(sender->silentSendInplace(d, num_ots, prng, chl));
    }

    void silent_receive_inplace(uint64_t num_ots) {
        if (receiver.get() == NULL) {
            throw std::runtime_error("receiver is null" LOCATION);
        }
        osuCrypto::cp::sync_wait(receiver->silentReceiveInplace(num_ots, prng, chl));
    }

    void get_sender_b(void* b, size_t n) {
        std::cout<<sender->mB.size()<<std::endl;
        if (n != sender->mB.size()) {
            throw std::runtime_error("size doesn't match" LOCATION);
        }
        memcpy(b, sender->mB.data(), sizeof(block) * n);
    }

    void get_receiver_c(void *c, size_t n) {
        if (n != receiver->mC.size()) {
            throw std::runtime_error("size doesn't match" LOCATION);
        }
        memcpy(c, receiver->mC.data(), sizeof(block) * n);
    }

    void get_receiver_a(void *a, size_t n) {
        if (n != receiver->mA.size()) {
            throw std::runtime_error("size doesn't match" LOCATION);
        }
        memcpy(a, receiver->mA.data(), sizeof(block) * n);
    }


    
}