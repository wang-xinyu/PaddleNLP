# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from tornado import web
from tornado import ioloop
import time
import urllib
import argparse
import chardet
import sys

import paddle
from faster_ernie_deploy_predict import Predictor
from paddlenlp.utils.log import logger

total_time = 0
# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default='./faster_infer_model/', help="The path to model parameters to be loaded.")
parser.add_argument("--batch_size", type=int, default=1, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"], help="The device to select to train the model, is must be cpu/gpu.")
parser.add_argument("--max_seq_length", type=int, default=60, help="The maximum total input sequence length after tokenization. ""Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--use_tensorrt', type=eval, default=False, choices=[True, False], help='Enable to use tensorrt to speed up.')
parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "int8"], help='The tensorrt precision.')
parser.add_argument('--cpu_threads', type=int, default=10, help='Number of threads to predict when using cpu.')
parser.add_argument('--enable_mkldnn', type=eval, default=False, choices=[True, False], help='Enable to use mkldnn to speed up when using cpu.')
parser.add_argument("--benchmark", type=eval, default=False, help="To log some information about environment and running.")
parser.add_argument("--save_log_path", type=str, default="./log_output/", help="The file path to save log.")
parser.add_argument("--test_ds_path", type=str, default="../../qt_rank_service/test_data", help="")
parser.add_argument("--server_port", type=int, default=8031, help="server port")
args = parser.parse_args()
# yapf: enable


def create_bert_server(args):
    predictor = Predictor(args.model_dir, args.device, args.max_seq_length,
                          args.batch_size, args.use_tensorrt, args.precision,
                          args.cpu_threads, args.enable_mkldnn)

    class BertHandler(web.RequestHandler):
        bert_predictor = predictor
        begin_time = -10

        def __init__(self, application, request, **kwargs):
            web.RequestHandler.__init__(self, application, request)

        def get(self):
            """
            Get request
            """
            results = self.http()
            self.write(json.dumps(results))

        def post(self):
            try:
                json_line = self.get_post_data()
                logger.info("json_line: {}".format(json_line, ))
                if json_line is None:
                    ret = {}
                    ret['status'] = 109
                    self.write(json.dumps(ret))
                    return
                input_data = json.loads(json_line)
                # ensure chinese character output
                result_str = self.get_proc_res(input_data)
                self.write(result_str)
            except Exception as e:
                ret = {}
                ret['status'] = 110
                self.write(json.dumps(ret))
                logger.error(str(e))

        def get_post_data(self):
            """Get http input data & decode into json object
            Returns:
                json object contains the input (q, p) paris
            """
            #logger.info("[get post data]")
            input_data = self.request.body
            #json_line = urllib.unquote(input_data)
            json_line = input_data
            if json_line is None or json_line == "":
                logger.error("Empty input - {}\n".format(input_data))
                return None
            return json_line

        def get_proc_res(self, input_data):
            """Data format transformation & call pretrained MRC model
            Args:
                Input json object contains the (q, p) paris
            Returns:
                The answer predicted by the MRC model (in json format)
            """
            #logger.info("[get_proc_res]")
            data_list = []
            qq_list = input_data['qq']
            for qq in qq_list:
                data_list.append([qq[0], qq[1]])
            self.begin_time = time.time()
            probs, idx = self.bert_predictor.predict(data_list)
            cost_time = time.time() - self.begin_time
            global total_time
            total_time += cost_time
            logger.info("predict total time:\t%f s, curr cost time:\t%f s" %
                        (total_time, cost_time))
            output_json = {}
            output_json['probability'] = []
            for prediction in probs:
                output_json['probability'].append(float(prediction[1]))
            output_json['status'] = 0
            result_str = json.dumps(output_json, ensure_ascii=False)
            #logger.info('[Output from model]: {}'.format(result_str))
            return result_str

    return BertHandler


def create_bert_app(sub_address, bert_server):
    """
    Create DQA server application
    """
    return web.Application([(sub_address, bert_server)])


if __name__ == "__main__":
    sub_address = r"/qq"
    bert_server = create_bert_server(args)
    app = create_bert_app(sub_address, bert_server)
    app.listen(args.server_port)
    ioloop.IOLoop.current().start()
