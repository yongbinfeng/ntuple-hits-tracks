#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <math.h>  

class GNN4TrackingProducer : public TritonEDProducer<> {
public:
  explicit GNN4TrackingProducer(edm::ParameterSet const& cfg)
      : TritonEDProducer<>(cfg, "GNN4TrackingProducer"),
        nnodes_(0),
        nedges_(0),
        threshold_(0.5),
        vnodes_(nullptr),
        vedges_(nullptr)
        {
  }

  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    int nodeMin_ = 1000;
    int nodeMax_ = 2000;
    int edgeMin_ = 10000;
    int edgeMax_ = 20000;
    std::uniform_int_distribution<int> randint1(nodeMin_, nodeMax_);
    nnodes_ = randint1(rng);
    std::uniform_int_distribution<int> randint2(edgeMin_, edgeMax_);
    nedges_ = randint2(rng);

    vedges_ = new std::vector<std::vector<int>>(nedges_, std::vector<int>(2, 0));
    // vnodes_ save the randomly generated deta and dphi information
    // not needed anymore in the real case
    vnodes_ = new std::vector<std::vector<float>>(nnodes_, std::vector<float>(2, 0));

    //set shapes
    auto& input0 = iInput.at("x__0");
    input0.setShape(0, nnodes_);
    auto data0 = input0.allocate<float>();
    auto& vdata0 = (*data0)[0];

    auto& input1 = iInput.at("edge_index__1");
    input1.setShape(1, nedges_);
    auto data1 = input1.allocate<int64_t>();
    auto& vdata1 = (*data1)[0];

    auto& input2 = iInput.at("edge_attr__2");
    input2.setShape(0, nedges_);
    auto data2 = input2.allocate<float>();
    auto& vdata2 = (*data2)[0];

    // fill in random eta, phi
    std::uniform_real_distribution<float> rand_eta(-4.0, 4.0);
    std::uniform_real_distribution<float> rand_phi(-3.14, 3.14);
    for (int i = 0; i < nnodes_; ++i) {
        vnodes_->at(i).at(0) = rand_eta(rng);
        vnodes_->at(i).at(1) = rand_phi(rng);
    }

    // fill in vdata
    std::normal_distribution<float> rand_x(-10, 4);
    for (unsigned i = 0; i < input0.sizeShape(); ++i) {
      vdata0.push_back(rand_x(rng));
    }

    std::uniform_int_distribution<int64_t> rand_edge_index(0, nnodes_ - 1);
    for (unsigned i = 0; i < input1.sizeShape(); ++i) {
      int inode = rand_edge_index(rng);
      vdata1.push_back(inode);
      vedges_->at(i/2).at(i%2) = inode;
    }

    std::normal_distribution<float> rand_edge_attr(1, 5);
    for (unsigned i = 0; i < input2.sizeShape(); ++i) {
      vdata2.push_back(rand_edge_attr(rng));
    }

    // convert to server format
    input0.toServer(data0);
    input1.toServer(data1);
    input2.toServer(data2);
  }
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    // check the results
    const auto& output0 = iOutput.begin()->second;
    // convert from server format
    const auto& tmp = output0.fromServer<float>();

    std::vector<std::vector<float>> distanceM(nnodes_, std::vector<float>(nnodes_, 100.0));
    for (int i = 0; i < nnodes_; ++i) {
        distanceM[i][i] = 0;
    }

    std::cout << "nedges " << nedges_ << " output shape " << output0.shape()[0] << std::endl;
    for (int i = 0; i < output0.shape()[0]/2; ++i) {
      // not sure yet why this is really needed
      float score = std::min(tmp[0][i], tmp[0][i + nedges_/2]);
      if (score > threshold_) {
        int inode = vedges_->at(i)[0];
        int onode = vedges_->at(i)[1];
        distanceM[inode][onode] = deltaR(inode, onode);
        std::cout << " deltaR between " << inode << " and " << onode << " is " << deltaR(inode, onode) << std::endl;
      }
    }
  }

  ~GNN4TrackingProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

private:
  float deltaR(int i, int j) {
    float dEta = vnodes_->at(i).at(0) - vnodes_->at(j).at(0);
    float dPhi = vnodes_->at(i).at(1) - vnodes_->at(j).at(1);
    if (dPhi > 3.1415926) 
      dPhi = dPhi - 3.1415926;
    else if (dPhi < -3.1415926) 
      dPhi = dPhi + 3.1415926;
    return sqrt(dEta*dEta + dPhi*dPhi);
  }

  int nnodes_;
  int nedges_;
  float threshold_;

  std::vector<std::vector<float>>* vnodes_;
  std::vector<std::vector<int>>* vedges_;

};

DEFINE_FWK_MODULE(GNN4TrackingProducer);
