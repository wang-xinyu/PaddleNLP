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
import numpy as np
import unittest
import paddle

__all__ = ['CommonTest', 'CpuCommonTest']


# assume all elements has same data type
def get_container_tpye(container):
    container_t = type(container)
    if container_t in [list, tuple]:
        if len(container) == 0:
            return container_t
        return get_container_tpye(container[0])
    return container_t


class CommonTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super(CommonTest, self).__init__(methodName=methodName)
        self.config = {}
        self.set_config()
        self.places = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.places.append('gpu')

    @classmethod
    def setUpClass(cls):
        '''
        Set test places for all test function
        '''
        for key, value in cls.__dict__.items():
            if key.startswith('test_'):
                value = CommonTest._test_places(value)
                setattr(cls, key, value)

    def _test_places(func):
        def wrapper(self, *args, **kw):
            places = self.places
            for place in places:
                paddle.set_device(place)
                try:
                    func(self, *args, **kw)
                except BaseException as ex:
                    raise Exception("{}, error in {} place.".format(ex, place))

        return wrapper

    def _check_output_impl(self, result, expected_result, equal=True):
        assertForNormalType = self.assertNotEqual
        assertForFloat = self.assertFalse
        if equal:
            assertForNormalType = self.assertEqual
            assertForFloat = self.assertTrue

        result_t = type(result)
        if result_t in [list, tuple]:
            result_t = get_container_tpye(result)

        if result_t in [str, int, bool, set]:
            assertForNormalType(result, expected_result)
        elif result_t in [float, np.array]:
            assertForFloat(np.allclose(result, expected_result))

    def check_output_equal(self, result, expected_result):
        self._check_output_impl(result, expected_result)

    def check_output_not_equal(self, result, expected_result):
        self._check_output_impl(result, expected_result, equal=False)

    def set_config(self):
        '''
        Set input arguments for tested api.
        '''
        pass


class CpuCommonTest(CommonTest):
    def __init__(self, methodName='runTest'):
        super(CpuCommonTest, self).__init__(methodName=methodName)
        self.places = ['cpu']
