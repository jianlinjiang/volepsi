#include "rvole.h"
#include <coproto/Socket/AsioSocket.h>
#include <libOTe/Tools/Coproto.h>
using OtExtSender = osuCrypto::SilentVoleSender;
using OtExtReceiver = osuCrypto::SilentVoleReceiver;
using Socket = osuCrypto::Socket;
using AsioSocket = coproto::AsioSocket;
using PRNG = osuCrypto::PRNG;
using block = osuCrypto::block;

enum Role {
    Sender = 1,
    Receiver = 2
};

std::unique_ptr<OtExtReceiver> receiver;
std::unique_ptr<OtExtSender> sender;
AsioSocket chl;
PRNG prng;
extern "C" {

    void init_silent_vole(int role, const char* ip, const uint64_t seed[2], uint64_t vector_size) {
        std::string ip_str(ip);
        block s(seed[0], seed[1]);
        prng.SetSeed(s);
        if (role == Role::Sender) {
            if (receiver.get() != NULL) {
                throw std::runtime_error("init should be called once" LOCATION);
            }
            sender = std::make_unique<OtExtSender>();
            chl = coproto::asioConnect(ip_str, true);
            sender->configure(vector_size, osuCrypto::SilentBaseType::Base);
        } else if (role == Role::Receiver ) {
            if (sender.get() != NULL) {
                throw std::runtime_error("init should be called once" LOCATION);
            }
            receiver = std::make_unique<OtExtReceiver>();
            chl = coproto::asioConnect(ip_str, true);
            receiver->configure(vector_size, osuCrypto::SilentBaseType::Base);
        } else {
            throw std::runtime_error("unkown role. " LOCATION);
        }
    }

    void silent_send_inplace(uint64_t delta[2], uint64_t vector_size) {
        block d(delta[0], delta[1]);
        osuCrypto::cp::sync_wait(sender->silentSendInplace(d, vector_size, prng, chl));
    }

    void silent_receive_inplace(uint64_t vector_size) {
        osuCrypto::cp::sync_wait(receiver->silentReceiveInplace(vector_size, prng, chl));
    }

}