/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "models/unigram.h"
#include <limits>
#include <sstream>

#include "utils/path.h"
#include "utils/unique_ptr.h"
#include "utils/utils.h"

namespace tokenizers {
namespace models {

Unigram::Unigram() {
  core::VocabList vocab = {{"<unk>", 0.0}};
  std::vector<size_t> unk_id = {0};
  Init(vocab, unk_id);
}

Unigram::Unigram(const core::VocabList& vocab,
                 const std::vector<size_t>& unk_id) {
  Init(vocab, unk_id);
}

void Unigram::Init(const core::VocabList& vocab,
                   const std::vector<size_t>& unk_id) {
  size_t n = vocab.size();
  if (unk_id.size() > 0) {
    if (n == 0) {
      std::ostringstream oss;
      oss << "EmptyVocabulary error occurs when init unigram with unk token.";
      throw std::runtime_error(oss.str());
    } else if (unk_id[0] >= n) {
      std::ostringstream oss;
      oss << "Unk token id is not in vocab when init unigram with unk token.";
      throw std::runtime_error(oss.str());
    }
  }

  vocab_ = vocab;
  unk_id_ = unk_id;

  size_t bos_id = n + 1;
  size_t eos_id = n + 2;
  double min_score = std::numeric_limits<double>::max();

  std::vector<const char*> keys;
  std::vector<int> values;

  for (size_t id = 0; id < n; ++id) {
    token_to_ids_.insert({vocab[id].first, id});
    keys.push_back(vocab[id].first.c_str());
    values.push_back(id);
    if (vocab[id].second < min_score) {
      min_score = vocab[id].second;
    }
  }

  std::vector<const char*> sorted_keys;
  std::vector<int> sorted_values;
  utils::GetSortedVocab(keys, values, &sorted_keys, &sorted_values);
  trie_ = utils::make_unique<Darts::DoubleArray>();
  if (trie_->build(sorted_keys.size(),
                   const_cast<char**>(&sorted_keys[0]),
                   nullptr,
                   &sorted_values[0]) != 0) {
    std::ostringstream oss;
    oss << "Cannot build double-array..";
    throw std::runtime_error(oss.str());
    return;
  }

  fuse_unk_ = true;
  is_optimized_ = true;
}

bool Unigram::TokenToId(const std::string& token, uint* id) const {
  if (token_to_ids_.find(token) == token_to_ids_.end()) {
    return false;
  }
  *id = token_to_ids_.at(token);
  return true;
}

bool Unigram::IdToToken(uint id, std::string* token) const {
  if (id >= vocab_.size()) {
    return false;
  }
  *token = vocab_[id].first;
  return true;
}

core::Vocab Unigram::GetVocab() const { return token_to_ids_; }

size_t Unigram::GetVocabSize() const { return vocab_.size(); }

std::vector<core::Token> Unigram::Tokenize(const std::string& sequence) {
  return {};
}

std::vector<std::string> Unigram::Save(
    const std::string& folder, const std::string& filename_prefix) const {
  std::string vocab_path;
  if (filename_prefix == "") {
    vocab_path = utils::PathJoin(folder, "unigram.json");
  } else {
    vocab_path = utils::PathJoin({folder, filename_prefix, "-unigram.json"});
  }
  VLOG(6) << "Vocab path" << vocab_path;
  std::ofstream fout(vocab_path);
  nlohmann::json j = *this;
  fout << j.dump();
  fout.close();
  return {vocab_path};
}


void to_json(nlohmann::json& j, const Unigram& model) {
  j = {{"type", "Unigram"}, {"unk_id", model.unk_id_}, {"vocab", model.vocab_}};
}

void from_json(const nlohmann::json& j, Unigram& model) {
  model.Init(j.at("vocab").get<core::VocabList>(),
             j.at("unk_id").get<std::vector<size_t>>());
}

}  // model
}  // tokenizers
