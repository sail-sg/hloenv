// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_SCHEDULE_H_
#define ALTGRAPH_SCHEDULE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "altgraph/hlo_module.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace altgraph {

#define CASE_NO_ARG_PASS(PASS_NAME, NAMESPACE)        \
  case PassType::PASS_NAME:                           \
    pass_ = std::make_unique<NAMESPACE::PASS_NAME>(); \
    break;

#define FIXED_ITERATION_LIMIT 25

class PassInterface {
 public:
  virtual ~PassInterface() = default;

  explicit PassInterface(const std::string name, int loop_count = 1)
      : name_(name), loop_count_(loop_count) {}

  const std::string& name() const { return name_; }

  bool changed() const { return changed_; }

  bool complete() const { return complete_; }

  int loop_count() const { return loop_count_; }

  virtual bool IsPipeline() = 0;

  bool Run(std::shared_ptr<AltHloModule> alt_hlo_module) {
    return Run(alt_hlo_module->hlo_module_ptr());
  }

  // Returns true if alternatives are generated, else false if a normal graph
  // is generated)
  bool Run(xla::HloModule* hlo_module) {
    // If loop_count_ == -1, run until there are no longer any changes
    // to the hlo_module, up to FIXED_ITERATION_LIMIT times
    bool fixed = loop_count_ == -1;
    int loop_count = fixed ? FIXED_ITERATION_LIMIT : loop_count_;

    if (complete_) {
      LOG(ERROR) << "Running pass/pipeline: " << name() << ", fixed = " << fixed
                 << ", loop_count_ = " << loop_count
                 << ", loop_iteration_count_ = " << loop_iteration_count_;
      complete_ = false;
      changed_ = false;
      loop_iteration_count_ = 0;
    } else {
      LOG(ERROR) << "    Continuing pass/pipeline: " << name();
    }

    bool inner_pass_has_alts;
    bool changed_this_run;
    bool pass_completed;
    for (int i = loop_iteration_count_; i < loop_count; i++) {
      RunHelperResults run_results = RunHelper(hlo_module);
      inner_pass_has_alts = run_results.generated_alts;
      changed_this_run = run_results.changed;
      pass_completed = run_results.completed;

      if (pass_completed) {
        loop_iteration_count_++;
        changed_ |= changed_this_run;
        complete_ = loop_iteration_count_ >= loop_count;
      }

      // Pass generated alts, hand control back to user to apply alts
      if (inner_pass_has_alts) {
        return true;
      }

      if (!changed_this_run && fixed) {
        break;
      }
      hlo_module->Cleanup();
    }

    complete_ = true;
    return false;  // If we get here, no alts were generated
  }

  struct RunHelperResults {
    bool generated_alts;
    bool changed;
    bool completed;
  };

 private:
  // Run the pass on the given HLO module.  Returns whether it generated alts
  // and whether it changed the hlo_module.
  virtual RunHelperResults RunHelper(xla::HloModule* hlo_module) = 0;

  const std::string name_;
  int loop_count_;

  int loop_iteration_count_ = 0;
  bool changed_ = false;
  bool complete_ = true;
};

class Pass : public PassInterface {
 public:
  explicit Pass(std::shared_ptr<xla::HloPassInterface> hlo_pass,
                int loop_count = 1)
      : PassInterface(std::string(hlo_pass->name()), loop_count),
        hlo_pass_(std::move(hlo_pass)) {}

  bool IsPipeline() override { return false; };

 private:
  RunHelperResults RunHelper(xla::HloModule* hlo_module) override {
    // Single pass never creates alternatives, that's done in the outer Run loop
    return {false, hlo_pass_->Run(hlo_module).ValueOrDie(), true};
  }

  std::shared_ptr<xla::HloPassInterface> hlo_pass_;
};

class Pipeline : public PassInterface {
 public:
  explicit Pipeline(const std::string pipeline_name, int loop_count)
      : PassInterface(pipeline_name, loop_count) {
    passes_ = std::vector<std::shared_ptr<PassInterface>>(0);
  }

  bool IsPipeline() override { return true; };

  void AddPass(std::shared_ptr<xla::HloPassInterface> hlo_pass, int count = 1) {
    std::shared_ptr<Pass> pass = std::make_shared<Pass>(hlo_pass, count);
    passes_.push_back(std::move(pass));
  }

  void AddPass(std::shared_ptr<PassInterface> pass) {
    passes_.push_back(std::move(pass));
  }

  void AddInvariantChecker(std::shared_ptr<xla::HloPassInterface> hlo_pass) {
    std::shared_ptr<Pass> pass =
        std::make_shared<Pass>(hlo_pass, /*loop_count*/ 1);
    invariant_checkers_.push_back(std::move(pass));
  }

  void AddInvariantChecker(std::shared_ptr<PassInterface> pass) {
    invariant_checkers_.push_back(std::move(pass));
  }

  void RunInvariantCheckers(xla::HloModule* hlo_module) {
    for (auto invariant_checker : invariant_checkers_) {
      invariant_checker->Run(hlo_module);
      if (invariant_checker->changed()) {
        LOG(FATAL) << "ERROR: Invariant checker must not change the graph!";
      }
    }
  }

  RunHelperResults RunHelper(xla::HloModule* hlo_module) override {
    // New run
    if (current_pass_idx_ == 0) {
      changed_this_run_ = false;
      RunInvariantCheckers(hlo_module);
      LOG(ERROR) << "Starting new run of Pipeline " << name();
    }

    bool completed = false;
    for (int pass_idx = current_pass_idx_; pass_idx < passes_.size();
         pass_idx++) {
      auto& pass = passes_.at(pass_idx);
      bool inner_pass_has_alts = pass->Run(hlo_module);

      if (pass->complete()) {
        changed_this_run_ |= pass->changed();
        current_pass_idx_ = pass_idx + 1;
        completed = current_pass_idx_ >= passes_.size();
      }
      if (inner_pass_has_alts) {
        // Hand control back to user to apply alts
        return {true, changed_this_run_, completed};
      }
    }
    current_pass_idx_ = 0;
    RunInvariantCheckers(hlo_module);
    return {false, changed_this_run_, true};
  }

  std::vector<std::shared_ptr<PassInterface>>& passes() { return passes_; }

 protected:
  bool changed_this_run_ = false;
  std::vector<std::shared_ptr<PassInterface>> passes_;

 private:
  int current_pass_idx_ = 0;
  std::vector<std::shared_ptr<PassInterface>> invariant_checkers_;
};

class AltPipeline : public Pipeline {
 public:
  explicit AltPipeline(std::shared_ptr<PassInterface> pass, int loop_count = 1)
      : Pipeline("alt_" + std::string(pass->name()), loop_count) {
    flatten_and_copy(std::move(pass));
  }

  void flatten_and_copy(std::shared_ptr<PassInterface> pass) {
    if (pass->IsPipeline()) {
      Pipeline* pipeline = dynamic_cast<Pipeline*>(pass.get());
      for (auto& sub_pass : pipeline->passes()) {
        flatten_and_copy(std::move(sub_pass));
      }
    } else {
      // Make a copy here, since we may want to run the original
      // pass/pipeline separate of this AltPipeline
      passes_.push_back(pass);
    }
  }

  RunHelperResults RunHelper(xla::HloModule* hlo_module) override {
    changed_this_run_ = false;
    hlo_module->SetDry(true);
    for (auto& pass : passes_) {
      pass->Run(hlo_module);
      changed_this_run_ |= pass->changed();
    }
    hlo_module->SetDry(false);
    bool generated_alts = changed_this_run_;
    return {generated_alts, changed_this_run_, true};
  }
};

}  // namespace altgraph

#endif  // ALTGRAPH_SCHEDULE_H_
