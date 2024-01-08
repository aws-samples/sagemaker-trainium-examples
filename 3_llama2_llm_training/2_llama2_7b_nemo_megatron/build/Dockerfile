# Base Docker image.
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04

RUN git clone -b 2.14_release https://github.com/aws-neuron/neuronx-nemo-megatron.git \
&& cd neuronx-nemo-megatron \
&& bash build.sh \
&& pip install build/*.whl \
&& pip install torch==1.13.* protobuf==3.20.3 \
&& pip install transformers==4.31.0 \
&& rm -fr /root/.cache/

RUN pip install Cython
RUN pip install pybind11

COPY requirements.txt /
RUN pip install -r requirements.txt

RUN cd /usr/local/lib/python3.10/site-packages/nemo/collections/nlp/data/language_modeling/megatron/ && make -C .
RUN python3 -c "from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import compile_helper;compile_helper()"
