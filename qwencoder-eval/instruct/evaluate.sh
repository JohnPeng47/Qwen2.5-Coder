MODEL_DIR=${1}
OUTPUT_DIR=${2}
TP=${3}
MODEL_DIR=${MODEL_DIR:-"./pretrained_models/"}
OUTPUT_DIR=${OUTPUT_DIR:-"./results/"}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/evalplus
mkdir -p ${OUTPUT_DIR}/livecodebench
mkdir -p ${OUTPUT_DIR}/MultiPL-E
mkdir -p ${OUTPUT_DIR}/bigcodebench
TP=${TP:-2}

ROOT_DIR="."
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"

vllm serve ${MODEL_DIR} \
    --served-model-name peft-codeqa-qwen \
    --tensor-parallel-size ${TP} \
    --trust-remote-code \
    --max-model-len 4096 \
    --dtype auto \
    --api-key token-abc123 \
    > ${OUTPUT_DIR}/vllm-server.txt 2>&1 &

cd ${ROOT_DIR}/eval-dev-quality
bash test.sh ${MODEL_DIR} ${TP} ${OUTPUT_DIR}/eval-dev-quality

# cd ${ROOT_DIR}/aider;
# bash test.sh ${MODEL_DIR} ${TP} ${OUTPUT_DIR}/aider

# cd ${ROOT_DIR}/multipl_e/chat;
# bash test.sh ${MODEL_DIR} ${TP} ${OUTPUT_DIR}/MultiPL-E

cd ${ROOT_DIR}/eval_plus;
bash test.sh ${MODEL_DIR} ${TP} ${OUTPUT_DIR}/evalplus

# cd ${ROOT_DIR}/BigCodeBench;
# bash test.sh ${MODEL_DIR} ${TP} ${OUTPUT_DIR}/bigcodebench

# cd ${ROOT_DIR}/cruxeval;
# bash test.sh ${MODEL_DIR} ${TP} ${OUTPUT_DIR}/cruxeval

cd ${ROOT_DIR}/livecode_bench;
bash test.sh ${MODEL_DIR} ${TP} ${OUTPUT_DIR}/livecodebench
