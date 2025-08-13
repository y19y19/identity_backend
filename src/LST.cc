// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <thread>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonbackend.h"

#include <alpaka/alpaka.hpp> // namespace alpaka
#include "../LSTCore/interface/LSTPrepareInput.h" // function lst::prepareInput 
#include "../LSTCore/interface/alpaka/LST.h" // class ALPAKA_ACCELERATOR_NAMESPACE::lst::LST
#include "../LSTCore/interface/TrackCandidatesHostCollection.h" // class lst::TrackCandidatesBaseHostCollection
#include "../LSTCore/interface/LSTInputHostCollection.h" // class lst::LSTInputHostCollection
#include "../LSTCore/interface/alpaka/TrackCandidatesDeviceCollection.h" // class ALPAKA_ACCELERATOR_NAMESPACE::lst::TrackCandidatesBaseDeviceCollection
#include "../LSTCore/interface/LSTInputHostCollection.h" // class lst::LstInputHostCollection
#include "../LSTCore/interface/alpaka/LSTInputDeviceCollection.h" // class ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTInputDeviceCollection
#include "../LSTCore/interface/LSTESData.h" // struct lst::LSTESData, function lst::loadAndFillESHost
#include "../LSTCore/interface/TrackCandidatesSoA.h" //class lst::TrackCandidatesBaseSoA

#include "HeterogeneousCore/AlpakaInterface/interface/config.h" 
// class alpaka_common::DevHost, aka alpaka::DevCpu
// ALPAKA_ACCELERATOR_NAMESPACE::Device, 
// ALPAKA_ACCELERATOR_NAMESPACE::Platform, 
// ALPAKA_ACCELERATOR_NAMESPACE::Queue
#include "DataFormats/Portable/interface/PortableCollection.h" 
// cms::alpakatools::CopyToDevice<PortableDeviceCollection<TLayout, TDevice>>::copyAsync
// cms::alpakatools::CopyToDevice<PortableHostCollection<TLayout>>::copyAsync
// class PortableCollection

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace backend { namespace LST {


// Define the GUARDED_RESPOND_IF_ERROR macro, to safely handle errors and send an error response back to the client.
/* If you're about to do something that might fail (like copying input buffers), 
wrap it in this macro. If it fails, Triton will send an error response to the client
and clean up properly — without crashing or leaking memory.
*/
#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

// Custom object to store global state for this backend
struct LSTBackendState {
  TRITONSERVER_MetricFamily* metric_family_ = nullptr;
  std::string message_ = "backend state";

  explicit LSTBackendState()
  {
#ifdef TRITON_ENABLE_METRICS
    // Create metric family
    THROW_IF_BACKEND_MODEL_ERROR(TRITONSERVER_MetricFamilyNew(
        &metric_family_, TRITONSERVER_METRIC_KIND_COUNTER,
        "input_byte_size_counter",
        "Cumulative input byte size of all requests received by the model"));
#endif  // TRITON_ENABLE_METRICS
  }

  ~LSTBackendState()
  {
#ifdef TRITON_ENABLE_METRICS
    if (metric_family_ != nullptr) {
      TRITONSERVER_MetricFamilyDelete(metric_family_);
    }
#endif  // TRITON_ENABLE_METRICS
  }
};

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  ~ModelState();

  // Get execution delay and delay multiplier
  uint64_t ExecDelay() const { return execute_delay_ms_; } // YY: assume this is not needed: 
  uint64_t DelayMultiplier() const { return delay_multiplier_; } // YY: assume this is not needed

  // Get the amount of nested custom trace spans to test
  bool EnableCustomTracing() const { return enable_custom_tracing_; } // YY: assume this is not needed
  /* YY: allows developers to track, record, and analyze the behavior and performance of model executions*/
  uint64_t NestedSpanCount() const { return nested_span_count_; } // YY: assume this is not needed
  uint64_t SingleActivityFrequency() const
  {
    return single_activity_frequency_; // YY: Could refer to how frequently a single kernel, operation, or step (e.g., data copy, inference execution) occurs.
  }

  const std::map<int, std::tuple<TRITONSERVER_DataType, std::vector<int64_t>>>&
  OptionalInputs()
  {
    return optional_inputs_; // YY: assume this is not needed
  }

  // Stores the instance count. Atomic to protect reads/writes by all instances.
  std::atomic<size_t> instance_count_;

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Block the thread for seconds specified in 'creation_delay_sec' parameter.
  // This function is used for testing.
  TRITONSERVER_Error* CreationDelay(); // YY: what is this? 

  // YY: initialize LST object
  lst::LST* fLST;

#ifdef TRITON_ENABLE_METRICS
  // Setup metrics for this backend.
  TRITONSERVER_Error* InitMetrics(
      TRITONSERVER_MetricFamily* family, std::string model_name,
      uint64_t model_version);
  // Update metrics for this backend.
  TRITONSERVER_Error* UpdateMetrics(uint64_t input_byte_size);
#endif  // TRITON_ENABLE_METRICS

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  // Delay time and multiplier to introduce into execution, in milliseconds.
  int execute_delay_ms_;
  int delay_multiplier_;

  // Amount of nested custom trace spans to test.
  bool enable_custom_tracing_{false};
  int nested_span_count_{0};
  int single_activity_frequency_{1};

  // Store index that the corresponding inputs can be optional. Also store
  // the output metadata to use, if an input is marked optional and not provided
  // in inference while the output is requested
  std::map<int, std::tuple<TRITONSERVER_DataType, std::vector<int64_t>>>
      optional_inputs_;

#ifdef TRITON_ENABLE_METRICS
  // Custom metrics associated with this model
  TRITONSERVER_Metric* input_byte_size_counter_ = nullptr;
#endif  // TRITON_ENABLE_METRICS
}; // end of class ModelState

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model, true /* allow_optional */), instance_count_(0),
      execute_delay_ms_(0), delay_multiplier_(0)
{
}

ModelState::~ModelState()
{
#ifdef TRITON_ENABLE_METRICS
  if (input_byte_size_counter_ != nullptr) {
    TRITONSERVER_MetricDelete(input_byte_size_counter_);
  }
#endif  // TRITON_ENABLE_METRICS
}

#ifdef TRITON_ENABLE_METRICS
TRITONSERVER_Error*
ModelState::InitMetrics(
    TRITONSERVER_MetricFamily* family, std::string model_name,
    uint64_t model_version)
{
  // Create labels for model/version pair to breakdown backend metrics per-model
  std::vector<const TRITONSERVER_Parameter*> labels;
  labels.emplace_back(TRITONSERVER_ParameterNew(
      "model", TRITONSERVER_PARAMETER_STRING, model_name.c_str()));
  labels.emplace_back(TRITONSERVER_ParameterNew(
      "version", TRITONSERVER_PARAMETER_STRING,
      std::to_string(model_version).c_str()));
  RETURN_IF_ERROR(TRITONSERVER_MetricNew(
      &input_byte_size_counter_, family, labels.data(), labels.size()));
  for (const auto label : labels) {
    TRITONSERVER_ParameterDelete(const_cast<TRITONSERVER_Parameter*>(label));
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::UpdateMetrics(uint64_t input_byte_size)
{
  RETURN_IF_ERROR(
      TRITONSERVER_MetricIncrement(input_byte_size_counter_, input_byte_size));
  return nullptr;  // success
}
#endif  // TRITON_ENABLE_METRICS

TRITONSERVER_Error*
ModelState::CreationDelay()
{
  // Feature for testing purpose...
  // look for parameter 'creation_delay_sec' in model config
  // and sleep for the value specified
  common::TritonJson::Value parameters;
  if (model_config_.Find("parameters", &parameters)) {
    common::TritonJson::Value creation_delay_sec;
    if (parameters.Find("creation_delay_sec", &creation_delay_sec)) {
      std::string creation_delay_sec_str;
      RETURN_IF_ERROR(creation_delay_sec.MemberAsString(
          "string_value", &creation_delay_sec_str));
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Creation delay is set to: ") + creation_delay_sec_str)
              .c_str());
      std::this_thread::sleep_for(
          std::chrono::seconds(std::stoi(creation_delay_sec_str)));
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig() // Removed most of the checks... Can ask people about what needs to be checked later. 
{
  // We have the json DOM for the model configuration...
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  /* YY: comment out because we do not need this check
  // There must be equal number of inputs and outputs.
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == outputs.ArraySize(), TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration must have equal input/output pairs"));
  */
  // Collect input/output names, shapes and datatypes
  std::map<std::string, std::tuple<std::string, std::vector<int64_t>>>
      input_infos, output_infos;

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("inputs array size:") + std::to_string(inputs.ArraySize())).c_str());  // inputs array size: 2, is the number of inputs?
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("outputs array size:") + std::to_string(outputs.ArraySize())).c_str());  
  

  // YY: comment out all the output check because it is not the same arraysize
  for (size_t input_index = 0; input_index < inputs.ArraySize(); input_index++) {

    common::TritonJson::Value input, output;
    RETURN_IF_ERROR(inputs.IndexAsObject(input_index, &input));
    //RETURN_IF_ERROR(outputs.IndexAsObject(io_index, &output)); // YY: outputs

    // Input and output names must follow INPUT/OUTPUT<index> pattern
    const char* input_name;
    size_t input_name_len;
    RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));

    /* YY: move this to another for loop for output to check that...
    const char* output_name;
    size_t output_name_len;
    RETURN_IF_ERROR(output.MemberAsString("name", &output_name, &output_name_len));
    */
 
    std::string input_name_str = std::string(input_name);
    /* YY: comment out because I do not use INPUT in the input name... no need for that
    RETURN_ERROR_IF_FALSE(
        input_name_str.rfind("INPUT", 0) == 0, TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "expected input name to follow INPUT<index> pattern, got '") +
            input_name + "'");
    */

    // Check if input is optional
    /* YY: no optional inputs are allowed in LST
    bool optional = false;
    RETURN_IF_ERROR(input.MemberAsBool("optional", &optional));
    */

    /* YY: comment out because I do not use OUTPUT in the output name... no need for that
    std::string output_name_str = std::string(output_name);
    RETURN_ERROR_IF_FALSE(
        output_name_str.rfind("OUTPUT", 0) == 0, TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "expected output name to follow OUTPUT<index> pattern, got '") +
            output_name + "'");
    */

    // Input and output must have same datatype
    std::string input_dtype; //, output_dtype;
    RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
    //RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype)); // YY: should be in another for loop 

    LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("YY: Validating Config: Checking input ") + input_name_str).c_str());

    // Input and output must have same shape or reshaped shape
    /* YY: comment out because we don't need to reshape using its ParseShape due to unflexible client... we can change client
    std::vector<int64_t> input_shape, output_shape;
    triton::common::TritonJson::Value reshape;
    if (input.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(backend::ParseShape(reshape, "shape", &input_shape));
    } else {
      RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));
    }

    if (output.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(backend::ParseShape(reshape, "shape", &output_shape));
    } else {
      RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));
    }
    */

    /*
    input_infos.insert(std::make_pair(
        input_name_str.substr(strlen("INPUT")),
        std::make_tuple(input_dtype, input_shape)));

    
    output_infos.insert(std::make_pair(
        output_name_str.substr(strlen("OUTPUT")),
        std::make_tuple(output_dtype, output_shape)));
    */

    /*
    if (optional) {
      const int idx = std::stoi(std::string(input_name, 5, -1));
      const auto dtype = ModelConfigDataTypeToTritonServerDataType(input_dtype);
      auto shape = input_shape;
      for (auto& dim : shape) {
        if (dim == -1) {
          dim = 1;
        }
      }
      optional_inputs_[idx] = std::make_tuple(dtype, shape);
    }
    */
  }

  // Must validate name, shape and datatype with corresponding input
  /* YY: comment out all the input check
  for (auto it = output_infos.begin(); it != output_infos.end(); ++it) {
    std::string output_index = it->first;
    auto input_it = input_infos.find(output_index);

    RETURN_ERROR_IF_FALSE(
        input_it != input_infos.end(), TRITONSERVER_ERROR_INVALID_ARG,
        std::string("expected input and output indices to match"));

    RETURN_ERROR_IF_FALSE(
        std::get<0>(input_it->second) == std::get<0>(it->second),
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("expected input and output datatype to match, got ") +
            std::get<0>(input_it->second) + " and " + std::get<0>(it->second));

    
    RETURN_ERROR_IF_FALSE(
        std::get<1>(input_it->second) == std::get<1>(it->second),
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("expected input and output shape to match, got ") +
            backend::ShapeToString(std::get<1>(input_it->second)) + " and " +
            backend::ShapeToString(std::get<1>(it->second)));
    
  }
  */

  /* YY: comment out because it is related to execute_delay, delay_multiplier, custom_tracing, nested_span and single_activity
  triton::common::TritonJson::Value params;
  if (model_config_.Find("parameters", &params)) {
    common::TritonJson::Value exec_delay;
    if (params.Find("execute_delay_ms", &exec_delay)) {
      std::string exec_delay_str;
      RETURN_IF_ERROR(
          exec_delay.MemberAsString("string_value", &exec_delay_str));
      execute_delay_ms_ = std::stoi(exec_delay_str);

      // Apply delay multiplier based on instance index, this is not taking
      // multiple devices into consideration, so the behavior is best controlled
      // in single device case.
      common::TritonJson::Value delay_multiplier;
      if (params.Find("instance_wise_delay_multiplier", &delay_multiplier)) {
        std::string delay_multiplier_str;
        RETURN_IF_ERROR(delay_multiplier.MemberAsString(
            "string_value", &delay_multiplier_str));
        delay_multiplier_ = std::stoi(delay_multiplier_str);
      }
    }
    common::TritonJson::Value custom_tracing;
    if (params.Find("enable_custom_tracing", &custom_tracing)) {
      std::string enable_custom_tracing_str;
      RETURN_IF_ERROR(custom_tracing.MemberAsString(
          "string_value", &enable_custom_tracing_str));
      enable_custom_tracing_ = enable_custom_tracing_str == "true";

      common::TritonJson::Value nested_span_count;
      if (params.Find("nested_span_count", &nested_span_count)) {
        std::string nested_span_count_str;
        RETURN_IF_ERROR(nested_span_count.MemberAsString(
            "string_value", &nested_span_count_str));
        nested_span_count_ = std::stoi(nested_span_count_str);

        common::TritonJson::Value single_activity_frequency;
        if (params.Find(
                "single_activity_frequency", &single_activity_frequency)) {
          std::string single_activity_frequency_str;
          RETURN_IF_ERROR(single_activity_frequency.MemberAsString(
              "string_value", &single_activity_frequency_str));
          single_activity_frequency_ = std::stoi(single_activity_frequency_str);
        }
      }
    }
  }
  */
  return nullptr;  // success
} // end ValidateModelConfig

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Get the instance ID of the instance.
  size_t InstanceId() const { return instance_id_; }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      const size_t instance_id);

  ModelState* model_state_;
  const size_t instance_id_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  // Must be initialized and incremented before passing to constructor
  // to guarantee unique_ids if parallel instance loading is supported.
  const auto instance_id = model_state->instance_count_++;
  try {
    *state =
        new ModelInstanceState(model_state, triton_model_instance, instance_id);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const size_t instance_id)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), instance_id_(instance_id)
{
}

/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments. This backend doesn't use
  // any such configuration but we print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // If we have any global backend state we create and set it here. We
  // make use of the global backend state here to track a custom metric across
  // all models using this backend if metrics are enabled.
  try {
    LSTBackendState* state = new LSTBackendState();
    RETURN_IF_ERROR(
        TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
} // end of TRITONBACKEND_Initialize

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  RETURN_ERROR_IF_TRUE(
      vstate == nullptr, TRITONSERVER_ERROR_INTERNAL,
      std::string("unexpected nullptr state in TRITONBACKEND_Finalize"));

  LSTBackendState* state = reinterpret_cast<LSTBackendState*>(vstate);
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: state is '") + state->message_ +
       "'")
          .c_str());

  delete state;
  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Can get location of the model artifacts. Normally we would need
  // to check the artifact type to make sure it was something we can
  // handle... but we are just going to log the location so we don't
  // need the check. We would use the location if we wanted to load
  // something from the model's repo.
  // YY: is it related to the .so files for model implementation? or just in general the model repo location? 
  TRITONBACKEND_ArtifactType artifact_type;
  const char* clocation;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &clocation));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Repository location: ") + clocation).c_str());

  // The model can access the backend as well... here we can access
  // the backend global state. We will use it to add per-model metrics
  // to the global metric family object stored in the state, if metrics
  // are enabled,
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  /* YY: why comment out?
  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  RETURN_ERROR_IF_TRUE(
      vbackendstate == nullptr, TRITONSERVER_ERROR_INTERNAL,
      std::string("unexpected nullptr state in TRITONBACKEND_ModelInitialize"));

  LSTBackendState* backend_state =
      reinterpret_cast<LSTBackendState*>(vbackendstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend state is '") + backend_state->message_ + "'")
          .c_str());
  */
  // Create a ModelState object and associate it with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  // For testing.. Block the thread for certain time period before returning.
  // YY: not sure whihc test needs delay...
  RETURN_IF_ERROR(model_state->CreationDelay());

#ifdef TRITON_ENABLE_METRICS
  // Create custom metric per model with metric family shared across backend
  RETURN_IF_ERROR(
      model_state->InitMetrics(backend_state->metric_family_, name, version));
#endif  // TRITON_ENABLE_METRICS

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
/* It means: f you use TRITONBACKEND_ModelSetState to store some custom 
state/data associated with the model (e.g., memory buffers, config objects, 
loaded weights), you are responsible for releasing that memory when the model 
is finalized.
*/
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  RETURN_ERROR_IF_TRUE(
      vstate == nullptr, TRITONSERVER_ERROR_INTERNAL,
      std::string("unexpected nullptr state in TRITONBACKEND_ModelFinalize"));

  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;
  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

#ifdef TRITON_ENABLE_GPU
  if (kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    cudaSetDevice(device_id); // YY: new thing in this branch
  }
#endif  // TRITON_ENABLE_GPU

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  RETURN_ERROR_IF_TRUE(
      vmodelstate == nullptr, TRITONSERVER_ERROR_INTERNAL,
      std::string(
          "unexpected nullptr state in TRITONBACKEND_ModelInstanceInitialize"));

  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

/* YY: comment out because we want to run on GPU
#ifndef TRITON_ENABLE_GPU
  // Because this backend just copies IN -> OUT and requires that
  // input and output be in CPU memory, we fail if a GPU instances is
  // requested.
  RETURN_ERROR_IF_FALSE(
      instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("'LST' backend only supports CPU instances"));
#endif  // TRITON_ENABLE_GPU
*/
  model_state->fLST = new lst::LST(); // YY: add LST executor
  return nullptr;  // success
} // end of ModelInstanceInitialize

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  RETURN_ERROR_IF_TRUE(
      vstate == nullptr, TRITONSERVER_ERROR_INTERNAL,
      std::string(
          "unexpected nullptr state in TRITONBACKEND_ModelInstanceFinalize"));

  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

#ifdef TRITON_ENABLE_GPU
  if (instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    cudaSetDevice(instance_state->DeviceId());
  }
#endif  // TRITON_ENABLE_GPU

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;
  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.

  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  RETURN_ERROR_IF_TRUE(
      instance_state == nullptr, TRITONSERVER_ERROR_INTERNAL,
      std::string(
          "unexpected nullptr state in TRITONBACKEND_ModelInstanceExecute"));
  ModelState* model_state = instance_state->StateForModel();

#ifdef TRITON_ENABLE_GPU
  if (instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    cudaSetDevice(instance_state->DeviceId()); // YY : it automatically assign Device Id?
  }
#endif // TRITON_ENABLE_GPU

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // 'responses' is initialized with the response objects below and
  // if/when an error response is sent the corresponding entry in
  // 'responses' is set to nullptr to indicate that that response has
  // already been sent.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  // Create a single response object for each request. If something
  // goes wrong when attempting to create the response objects just
  // fail all of the requests by returning an error.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  // After this point we take ownership of 'requests', which means
  // that a response must be sent for every request. If something does
  // go wrong in processing a particular request then we send an error
  // response just for the specific request.

  for (uint32_t r = 0; r < request_count; ++r) {

    TRITONBACKEND_Request* request = requests[r];

    // YY: ignore this if condition
    if (model_state->EnableCustomTracing()) { 
      // Example for tracing a custom activity from the backend
      TRITONSERVER_InferenceTrace* trace;
      GUARDED_RESPOND_IF_ERROR(
          responses, r, TRITONBACKEND_RequestTrace(request, &trace));

      auto nesting_count = model_state->NestedSpanCount();
      auto singl_act_frequency = model_state->SingleActivityFrequency();

      if (trace != nullptr) {
        uint64_t custom_activity_ns;
        const char* activity_name = "CUSTOM_ACTIVITY_START";

        SET_TIMESTAMP(custom_activity_ns);
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_InferenceTraceReportActivity(
                trace, custom_activity_ns, activity_name));

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        activity_name = "CUSTOM_ACTIVITY_END";
        SET_TIMESTAMP(custom_activity_ns);
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_InferenceTraceReportActivity(
                trace, custom_activity_ns, activity_name));


        uint64_t custom_single_activity_ns;
        const char* single_activity_name = "CUSTOM_SINGLE_ACTIVITY";
        SET_TIMESTAMP(custom_single_activity_ns);
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_InferenceTraceReportActivity(
                trace, custom_single_activity_ns, single_activity_name));


        for (uint32_t count = 0; count < nesting_count; count++) {
          std::string start_span_str =
              std::string("CUSTOM_ACTIVITY" + std::to_string(count) + "_START");
          SET_TIMESTAMP(custom_activity_ns);
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONSERVER_InferenceTraceReportActivity(
                  trace, custom_activity_ns, start_span_str.c_str()));
          std::this_thread::sleep_for(std::chrono::milliseconds(100));

          if (count % singl_act_frequency == 0) {
            SET_TIMESTAMP(custom_single_activity_ns);
            GUARDED_RESPOND_IF_ERROR(
                responses, r,
                TRITONSERVER_InferenceTraceReportActivity(
                    trace, custom_single_activity_ns, single_activity_name));
          }
        }

        for (uint32_t count = 0; count < nesting_count; count++) {
          std::string end_span_str = std::string(
              "CUSTOM_ACTIVITY" + std::to_string(nesting_count - count - 1) +
              "_END");
          SET_TIMESTAMP(custom_activity_ns);
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONSERVER_InferenceTraceReportActivity(
                  trace, custom_activity_ns, end_span_str.c_str()));
        }
      }
    } // end enable custom tracing

    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestId(request, &request_id));

    uint32_t input_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInputCount(request, &input_count));

    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read request input/output counts, error response sent")
              .c_str());
      continue;
    }

    LOG_MESSAGE(
      TRITONSERVER_LOG_ERROR,
      (std::string("requested input count: ") + std::to_string(input_count))
          .c_str()); // YY
    
    // For statistics we need to collect the total batch size of all the
    // requests. If the model doesn't support batching then each request is
    // necessarily batch-size 1. If the model does support batching then the
    // first dimension of the shape is the batch size. We only the first input
    // for this.

    // We validated that the model configuration specifies N inputs, but the
    // request is not required to request any output at all so we only produce
    // outputs that are requested.
    LOG_MESSAGE(
      TRITONSERVER_LOG_ERROR,
      (std::string("requested output count: ") + std::to_string(requested_output_count))
          .c_str()); // YY: it should be 1, double check!!

    std::map<std::string, const void*> inputs_name_buffer;
    std::map<std::string, uint32_t> inputs_name_buffer_byte_size;
    std::vector<TRITONBACKEND_Input*> inputs_ptr(input_count, nullptr);
    std::vector<TRITONSERVER_DataType> inputs_datatype(input_count);
    std::vector<int64_t*> inputs_shape(input_count, nullptr);
    std::vector<uint32_t> inputs_dims_count(input_count);
    std::vector<uint32_t> inputs_byte_size(input_count);
    std::vector<uint32_t> inputs_buffer_count(input_count);
    std::vector<const void*> inputs_buffer_ptr(input_count, nullptr);
    std::vector<uint64_t> inputs_buffer_byte_size(input_count, 0); // YY: Is it the same as inputs_byte_size?
    std::vector<TRITONSERVER_MemoryType> inputs_memory_type(input_count,TRITONSERVER_MEMORY_CPU);
    std::vector<int64_t> inputs_memory_type_id(input_count, 0);

    for (uint32_t input_index = 0; input_index < input_count; input_index++) {

      // find input name 
      const char* input_name;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestInputName(request, input_index, &input_name));

      // find input that corresponding to the name 
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestInput(request, input_name.c_str(), &inputs_ptr[input_index]));

      // If an error response was sent while getting the input then display an
      // error message and move on to next request.
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to read input 0, error response sent")
                .c_str());
        continue;
      }
      
      // find metadata, include datatype, input shape, input dimension, input byte, and input buffer count
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_InputProperties(
              inputs_ptr[input_index], &input_name, &inputs_datatype[input_index], &inputs_shape[input_index],
              &inputs_dims_count[input_index], &inputs_byte_size[input_index], &inputs_buffer_count[input_index])); 
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to read input properties, error response sent")
                .c_str());
        continue;
      }

      // find input buffer
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_InputBuffer(
              inputs_ptr[input_index], 0, &inputs_buffer_ptr[input_index], &inputs_buffer_byte_size[input_index], &inputs_memory_type[input_index],
              &inputs_memory_type_id[input_index])); // The 0 here is input buffer count
      if (responses[r] == nullptr) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "failed to get input buffer"));
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to get input buffer, error response sent")
                .c_str());
        continue;
      }

      // keep a copy of map, mapping input name with pointer to input buffer
      inputs_name_buffer[str(input_name)] = inputs_buffer_ptr[input_index];
      inputs_name_buffer_byte_size[str(input_name)] = inputs_buffer_byte_size[input_index];
    } // end of loop for input_index


    // Just for printing out purpose if you want to check the inputs:
    // Cast buffer into the type that the model's operation function takes
    /*
    const float* input0_floatbuffer = static_cast<const float*>(inputs_buffer_ptr[0]);

    LOG_MESSAGE( // YY: modify this if needed
      TRITONSERVER_LOG_ERROR,
      (std::string("input_0: ") + std::to_string(input0_floatbuffer[0]) + std::string(" input buffer byte size: ") + std::to_string(inputs_buffer_byte_size[0]) )
          .c_str());
    */

    // Run LST algorithm
    // YY: need to implement the convertion of std::vector<std::vector<int>>
    // YY: need to call prepareInput here
    // YY: need to load ESdata
    // YY: need to copy input to device here
    // YY: need to call LST->Run() here
    
    const float* see_px_ptr = static_cast<const float*>(inputs_name_buffer["see_px"]);
    const float* see_py_ptr = static_cast<const float*>(inputs_name_buffer["see_py"]);
    const float* see_pz_ptr = static_cast<const float*>(inputs_name_buffer["see_pz"]);
    const float* see_dxy_ptr = static_cast<const float*>(inputs_name_buffer["see_dxy"]);
    const float* see_dz_ptr = static_cast<const float*>(inputs_name_buffer["see_dz"]);
    const float* see_ptErr_ptr = static_cast<const float*>(inputs_name_buffer["see_ptErr"]);
    const float* see_etaErr_ptr = static_cast<const float*>(inputs_name_buffer["see_etaErr"]);
    const float* see_stateTrajGlbX_ptr = static_cast<const float*>(inputs_name_buffer["see_stateTrajGlbX"]);
    const float* see_stateTrajGlbY_ptr = static_cast<const float*>(inputs_name_buffer["see_stateTrajGlbY"]);
    const float* see_stateTrajGlbZ_ptr = static_cast<const float*>(inputs_name_buffer["see_stateTrajGlbZ"]);
    const float* see_stateTrajGlbPx_ptr = static_cast<const float*>(inputs_name_buffer["see_stateTrajGlbPx"]);
    const float* see_stateTrajGlbPy_ptr = static_cast<const float*>(inputs_name_buffer["see_stateTrajGlbPy"]);
    const float* see_stateTrajGlbPz_ptr = static_cast<const float*>(inputs_name_buffer["see_stateTrajGlbPz"]);
    const int* see_q_ptr = static_cast<const int*>(inputs_name_buffer["see_q"]);
    const int* see_hit_size_ptr = static_cast<const int*>(inputs_name_buffer["see_hit_size"]);
    const int* see_hitIdx_ptr = static_cast<const int*>(inputs_name_buffer["see_hitIdx"]);
    const unsigned int* see_algo_ptr = static_cast<const unsigned int*>(inputs_name_buffer["see_algo"]);
    const unsigned int* ph2_detId_ptr = static_cast<const unsigned int*>(inputs_name_buffer["ph2_detId"]);
    const float* ph2_x_ptr = static_cast<const float*>(inputs_name_buffer["ph2_x"]);
    const float* ph2_y_ptr = static_cast<const float*>(inputs_name_buffer["ph2_y"]);
    const float* ph2_z_ptr = static_cast<const float*>(inputs_name_buffer["ph2_z"]);
    const float* ptCut_ptr = static_cast<const float*>(inputs_name_buffer["ptCut"]);

    int N_seed = inputs_name_buffer_byte_size["see_px"]/sizeof(float);
    std::vector<float> see_px(see_px_ptr, see_px_ptr + N_seed);
    std::vector<float> see_py(see_py_ptr, see_py_ptr + N_seed);
    std::vector<float> see_pz(see_pz_ptr, see_pz_ptr + N_seed);
    std::vector<float> see_dxy(see_dxy_ptr, see_dxy_ptr + N_seed);
    std::vector<float> see_dz(see_dz_ptr, see_dz_ptr + N_seed);
    std::vector<float> see_ptErr(see_ptErr_ptr, see_ptErr_ptr + N_seed);
    std::vector<float> see_etaErr(see_etaErr_ptr, see_etaErr_ptr + N_seed);
    std::vector<float> see_stateTrajGlbX(see_stateTrajGlbX_ptr, see_stateTrajGlbX_ptr + N_seed);
    std::vector<float> see_stateTrajGlbY(see_stateTrajGlbY_ptr, see_stateTrajGlbY_ptr + N_seed);
    std::vector<float> see_stateTrajGlbZ(see_stateTrajGlbZ_ptr, see_stateTrajGlbZ_ptr + N_seed);
    std::vector<float> see_stateTrajGlbPx(see_stateTrajGlbPx_ptr, see_stateTrajGlbPx_ptr + N_seed);
    std::vector<float> see_stateTrajGlbPy(see_stateTrajGlbPy_ptr, see_stateTrajGlbPy_ptr + N_seed);
    std::vector<float> see_stateTrajGlbPz(see_stateTrajGlbPz_ptr, see_stateTrajGlbPz_ptr + N_seed);

    std::vector<int> see_q(see_q_ptr, see_q_ptr + N_seed);
    std::vector<int> see_hit_size(see_hit_size_ptr, see_hit_size_ptr + N_seed);
    int N_hit = 0;
    for (int number: see_hit_size) N_hit += number;
    std::vector<int> see_hitIdx_flat(see_hitIdx_ptr, see_hitIdx_ptr + N_hit);
    std::vector<std::vector<int>> see_hitIdx;
    size_t index = 0;
    for (int size : see_hit_size) {
        see_hitIdx.emplace_back(see_hitIdx_flat.begin() + index, see_hitIdx_flat.begin() + index + size);
        index += size;
    }
    std::vector<unsigned int> see_algo(see_algo_ptr, see_algo_ptr + N_seed);
    int N_ph2 = inputs_name_buffer_byte_size["ph2_x"]/sizeof(float);
    std::vector<unsigned int> ph2_detId(ph2_detId_ptr, ph2_detId_ptr + N_ph2);
    std::vector<float> ph2_x(ph2_x_ptr, ph2_x_ptr + N_ph2);
    std::vector<float> ph2_y(ph2_y_ptr, ph2_y_ptr + N_ph2);
    std::vector<float> ph2_z(ph2_z_ptr, ph2_z_ptr + N_ph2);
    float ptCut = *ptCut_ptr;

    // YY: to use the prepareInput function... it needs a Queue which is not necessary... can I avoid it in some way?
    ALPAKA_ACCELERATOR_NAMESPACE::Device devAcc = alpaka::getDevByIdx(ALPAKA_ACCELERATOR_NAMESPACE::Platform{}, 0u); 
    std::vector<ALPAKA_ACCELERATOR_NAMESPACE::Queue> queues; 
    queues.push_back(ALPAKA_ACCELERATOR_NAMESPACE::Queue(devAcc));

    lst::LSTInputHostCollection lstInputHC = 
      prepareInput(see_px,
                   see_py,
                   see_pz,
                   see_dxy,
                   see_dz,
                   see_ptErr,
                   see_etaErr,
                   see_stateTrajGlbX,
                   see_stateTrajGlbY,
                   see_stateTrajGlbZ,
                   see_stateTrajGlbPx,
                   see_stateTrajGlbPy,
                   see_stateTrajGlbPz,
                   see_q,
                   see_hitIdx,
                   see_algo,
                   ph2_detId,
                   ph2_x,
                   ph2_y,
                   ph2_z,
                   ptCut,
                   queues[0]);

    // Get ESData on Host and then copy to Device
    std::string ptCutString = (ptCut >= 0.8) ? "0.8" : "0.6";
    std::unique_ptr<lst::LSTESData<alpaka_common::DevHost>> hostESData = lst::loadAndFillESHost(ptCutString); 
    lst::LSTESData<ALPAKA_ACCELERATOR_NAMESPACE::Device> const* deviceESData = 
      cms::alpakatools::CopyToDevice<lst::LSTESData<alpaka_common::DevHost>>::copyAsync(queues[0], *hostESData.get()); 
    // Copy input from Host to Device
    ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTInputDeviceCollection lstInputDC(lstInputHC.sizes(), queues[0]); 
    alpaka::memcpy(queues[0], lstInputDC->buffer(), lstInputHC->buffer());
    alpaka::wait(queues[0]);
    // Run LST 
    model_state->fLST->run(queues[0], 
                           false /*verbose*/, 
                           ptCut, 
                           deviceESData, 
                           lstInputDC, 
                           false /*no_pls_dupclean*/,
                           false /*tc_pls_triplets*/, 
                           );

/* YY: why comment out? 
#ifdef TRITON_ENABLE_METRICS
    GUARDED_RESPOND_IF_ERROR(
        responses, r, model_state->UpdateMetrics(input_byte_size));
#endif  // TRITON_ENABLE_METRICS
*/

    // YY: need to get Output here
    // YY: need to copy output to host here
    // YY: Setting output_tmp, this is unregistered location
    //const void* output_tmp = model_state->fLST->getTrackCandidates(); 

    // Get the output  
    std::unique_ptr<ALPAKA_ACCELERATOR_NAMESPACE::lst::TrackCandidatesBaseDeviceCollection> trackCandidatesBaseDC_ = model_state->fLST->getTrackCandidates(); 
    // Copy output to host
    lst::TrackCandidatesBaseHostCollection trackCandidatesBaseHC_;
    // YY: why I cannot copy directly like before? 
    //alpaka::memcpy(queues[0], lstInputHC->buffer(), lstInputDC->buffer());

    trackCandidatesBaseHC_.emplace(
          cms::alpakatools::CopyToHost<::PortableCollection<TrackCandidatesBaseSoA, devAcc>>::copyAsync(
              queues[0], *trackCandidatesBaseDC_)); // maybe the devAcc is not right for TDev? and PortableCollection is included?
    // Get the pointer to the raw buffer
    const uint8_t* output_byte_ptr = reinterpret_cast<const uint8_t*>(trackCandidatesBaseHC_.data());
    const void* output_raw_buffer_ptr = output_byte_ptr;
    // Get the byte size of the output
    uint64_t output_buffer_byte_size = trackCandidatesBaseHC_.byteSize(); // YY: should test first if this works in normal code in lst.cc

    // YY: manually set output dimension. We know that it will be 1D flat Raw Bytes
    int64_t* output_shape = new int64_t[1];
    output_shape[0] = output_buffer_byte_size; // uint8 is 1 byte
    uint32_t output_dims_count = 1;

    const char* output_name = nullptr;
    TRITONBACKEND_Output* output;
    void* output_buffer;
    TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t output_memory_type_id = 0; // YY: 0 is on CPU... 

    LOG_MESSAGE(
      TRITONSERVER_LOG_ERROR,
      (std::string("output_buffer_byte_size: ") + std::to_string(output_buffer_byte_size))
          .c_str());

    // Request OutputName
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputName(
            request, 0 /* index */, &output_name));

    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read requested output name, error response sent")
              .c_str());
      continue;
    }

    // Request Output
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_ResponseOutput(
            responses[r], &output, output_name, TRITONSERVER_TYPE_UINT8, output_shape,
            output_dims_count)); // YY: set datatype... maybe this is flexible in the config.pbtxt.

    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to create response output, error response sent")
              .c_str());
      continue;
    }

    // Set Output buffer to map to output
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_OutputBuffer(
            output, &output_buffer, output_buffer_byte_size, &output_memory_type,
            &output_memory_type_id));

    if ((responses[r] == nullptr) ||
          (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "failed to create output buffer in CPU memory")); // YY: does output buffer has to be on CPU? not GPU? the new sample do not have this check
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to create output buffer in CPU memory, error response "
           "sent")
              .c_str());
      continue;
    }
    
    // Copy output to the registered output_buffer... 
    // YY: Maybe it can be optimized to not copy? This is moving on the Host/CPU.
    memcpy(output_buffer,output_raw_buffer_ptr,output_buffer_byte_size); 

    /* YY: just for print out the output. Since the output fill be raw bytes, there is no need to print out
    const float* output_floatbuffer = static_cast<const float*>(output_tmp);
    LOG_MESSAGE(
      TRITONSERVER_LOG_ERROR,
      (std::string("output_tmp: ") + std::to_string(output_floatbuffer[0]))
          .c_str());
    */

  } // end loop for request_count


/* YY: comment this out because I don't know what cuda_copy and cudaStreamSynchronize do
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(instance_state->CudaStream());
  }
#endif  // TRITON_ENABLE_GPU
*/

  // If we get to this point then there hasn't been any error and the response
  // is complete and we can send it. This is the last (and only) response that
  // we are sending for the request so we must mark it FINAL. If there is an
  // error when sending all we can do is log it.
  for (auto& response : responses) { 
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed sending response");
    }
  }

  // There are two types of statistics that we can report... the statistics for
  // the entire batch of requests that we just executed and statistics for each
  // individual request. Here we report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    //LOG_IF_ERROR( // YY: comment out instance report
    //    TRITONBACKEND_ModelInstanceReportStatistics(
    //        instance_state->TritonModelInstance(), request,
    //        (responses[r] != nullptr) /* success */, exec_start_ns,
    //        compute_start_ns, compute_end_ns, exec_end_ns),
    //    "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request"); 
  }

  /* // YY: comment out batch report
  // Here we report statistics for the entire batch of requests.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_state->TritonModelInstance(), total_batch_size,
          exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");
  */
  return nullptr;  // success

} // end of ModelInstanceExecute

TRITONSERVER_Error*
TRITONBACKEND_GetBackendAttribute(
    TRITONBACKEND_Backend* backend,
    TRITONBACKEND_BackendAttribute* backend_attributes)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_GetBackendAttribute: setting attributes");
  // This backend can safely handle parallel calls to
  // TRITONBACKEND_ModelInstanceInitialize (thread-safe).
  RETURN_IF_ERROR(TRITONBACKEND_BackendAttributeSetParallelModelInstanceLoading(
      backend_attributes, true));

  return nullptr;
} // end of GetBackendAttribute
}  // extern "C"

}}}  // namespace triton::backend::LST

