"""Portable single-model configuration. No internal registry or default model path."""

from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.longbench.longbench import longbench_datasets

os = __import__("os")
torch = __import__("torch")
__import__("model_adapter")
NoThinkingChatModel = "model_adapter.NoThinkingChatModel"
graphwalks = __import__("graphwalks_components")
GraphWalks32KDataset, GraphWalksEvaluator = "graphwalks_components.GraphWalks32KDataset", "graphwalks_components.GraphWalksEvaluator"
longreason = __import__("longreason_components")
LongReasonLocalDataset, LongReasonEvaluator = "longreason_components.LongReasonLocalDataset", "longreason_components.LongReasonEvaluator"
babilong = __import__("babilong_components")
BabiLongLocalDataset, BabiLongDualEvaluator, TASKS = "babilong_components.BabiLongLocalDataset", "babilong_components.BabiLongDualEvaluator", babilong.TASKS

benchmark = os.environ["LCCP_BENCHMARK"]
model_path = os.environ["LCCP_MODEL"]
max_output = {"longbench": 512, "graphwalks": 128, "longreason": 512, "babilong": 20}[benchmark]
max_prompt = {"longbench": 32256, "graphwalks": 32256, "longreason": 32256, "babilong": 32768}[benchmark]
models = [dict(
    type=NoThinkingChatModel, abbr="candidate", path=model_path,
    max_out_len=max_output, max_seq_len=max_prompt,
    batch_size=int(os.environ.get("LCCP_BATCH_SIZE", "1")),
    tokenizer_kwargs=dict(trust_remote_code=True, local_files_only=True),
    model_kwargs=dict(torch_dtype="torch.bfloat16", trust_remote_code=True, local_files_only=True,
                      tp_plan=None,
                      attn_implementation=os.environ.get("LCCP_ATTENTION", "flash_attention_2")),
    generation_kwargs=dict(do_sample=False), run_cfg=dict(num_gpus=1),
)]
infer_cfg = dict(
    prompt_template=dict(type="opencompass.openicl.icl_prompt_template.PromptTemplate",
                         template=dict(round=[dict(role="HUMAN", prompt="{prompt}")])),
    retriever=dict(type="opencompass.openicl.icl_retriever.ZeroRetriever"),
    inferencer=dict(type="opencompass.openicl.icl_inferencer.GenInferencer", max_out_len=max_output),
)
if benchmark == "longbench":
    datasets = longbench_datasets
elif benchmark == "graphwalks":
    datasets = [dict(
        abbr="graphwalks_" + problem, type=GraphWalks32KDataset, path=os.environ["GRAPHWALKS_DATA"],
        tokenizer_path=model_path, problem_type=problem, data_file="graphwalks_128k_and_shorter.parquet",
        max_rendered_tokens=32000, reader_cfg=dict(input_columns=["prompt"], output_column="answer"),
        infer_cfg=infer_cfg, eval_cfg=dict(evaluator=dict(type=GraphWalksEvaluator),
                                         pred_postprocessor=dict(type="graphwalks_strict")),
    ) for problem in ["bfs", "parents"]]
elif benchmark == "longreason":
    datasets = [dict(
        abbr="longreason_32k", type=LongReasonLocalDataset, path=os.environ["LONGREASON_DATA"], split="32k",
        reader_cfg=dict(input_columns=["prompt"], output_column="answer"), infer_cfg=infer_cfg,
        eval_cfg=dict(evaluator=dict(type=LongReasonEvaluator), pred_role="BOT"),
    )]
else:
    datasets = [dict(
        abbr="babilong_" + task + "_32k", type=BabiLongLocalDataset, path=os.environ["BABILONG_DATA"],
        task=task, split_name="32k", reader_cfg=dict(input_columns=["prompt"], output_column="reference"),
        infer_cfg=infer_cfg, eval_cfg=dict(evaluator=dict(type=BabiLongDualEvaluator, task=task)),
    ) for task in TASKS]

del os, torch, graphwalks, longreason, babilong
