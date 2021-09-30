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

class GNN4TrackingProducer : public TritonEDProducer<> {
public:
  explicit GNN4TrackingProducer(edm::ParameterSet const& cfg)
      : TritonEDProducer<>(cfg, "GNN4TrackingProducer")
        {
  }

  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    std::mt19937 rng(0);
    int nodeMin_ = 1000;
    int nodeMax_ = 2000;
    int edgeMin_ = 10000;
    int edgeMax_ = 10000;
    std::uniform_int_distribution<int> randint1(nodeMin_, nodeMax_);
    int nnodes = randint1(rng);
    std::uniform_int_distribution<int> randint2(edgeMin_, edgeMax_);
    int nedges = randint2(rng);

    //set shapes
    auto& input0 = iInput.at("x__0");
    input0.setShape(0, nnodes);
    auto data0 = input0.allocate<float>();
    auto& vdata0 = (*data0)[0];

    auto& input1 = iInput.at("edge_index__1");
    input1.setShape(1, nedges);
    auto data1 = input1.allocate<int64_t>();
    auto& vdata1 = (*data1)[0];

    auto& input2 = iInput.at("edge_attr__2");
    input2.setShape(0, nedges);
    auto data2 = input2.allocate<float>();
    auto& vdata2 = (*data2)[0];

    // fill in vdata
    std::normal_distribution<float> rand_x(-10, 4);
    for (unsigned i = 0; i < input0.sizeShape(); ++i) {
      vdata0.push_back(rand_x(rng));
    }

    std::uniform_int_distribution<int64_t> rand_edge_index(0, nnodes - 1);
    for (unsigned i = 0; i < input1.sizeShape(); ++i) {
      vdata1.push_back(rand_edge_index(rng));
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
    const auto& output1 = iOutput.begin()->second;
    // convert from server format
    const auto& tmp = output1.fromServer<float>();
    for (int i = 0; i < output1.shape()[0]; ++i) {
      std::cout << "output " << i << ": ";
      for (int j = 0; j < output1.shape()[1]; ++j) {
          std::cout << tmp[0][output1.shape()[1] * i + j] << " ";
      }
      std::cout << std::endl;
    }
  }
  ~GNN4TrackingProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

};

DEFINE_FWK_MODULE(GNN4TrackingProducer);
