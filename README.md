# Evaluating-Socio-Ecological-Bias-in-Retrieval-Augmented-Generation
This project provides a controlled evaluation framework for investigating whether disciplinary imbalances in socio-ecological knowledge propagate into Retrieval-Augmented Generation (RAG) systems. 
The Problem
RAG systems are increasingly used for evidence synthesis in complex scientific domains, but their reliability depends on representing interdisciplinary knowledge without systematically favoring one domain. Prior research has documented historical emphasis on social over ecological perspectives in socio-ecology, and we hypothesize this imbalance may propagate into RAG systems.

Methodology
We conduct two controlled experiments across five open-weight LLMs (GPT-OSS, Llama-3.2, Mistral, Qwen3-VL, DeepSeek-R1) and two embedding models (Nomic-Embed-Text, EmbeddingGemma):

Retrieval-Bias Experiment: Vary the retrieval corpus (social vs. ecological) while holding prompts fixed

Prompt-Bias Experiment: Hold the retrieval corpus fixed while varying semantically equivalent prompt formulations

Evaluation uses a dual-agent LLM-as-judge framework measuring Answer Relevance, Faithfulness, and Context Relevance.
