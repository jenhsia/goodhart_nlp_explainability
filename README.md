# [Goodhart's Law Applies to NLP's Explanation Benchmarks](https://arxiv.org/abs/2308.14272)

## Description
In this paper, we critically examine two sets of metrics: the ERASER metrics (comprehensiveness and sufficiency) and the EVAL-X metrics, focusing our inquiry on natural language processing. First, we show that we can inflate a model's comprehensiveness and sufficiency scores dramatically without altering its predictions or explanations on in-distribution test inputs. Our strategy exploits the tendency for extracted explanations and their complements to be "out-of-support" relative to each other and in-distribution inputs. Next, we demonstrate that the EVAL-X metrics can be inflated arbitrarily by a simple method that encodes the label, even though EVAL-X is precisely motivated to address such exploits. Our results raise doubts about the ability of current metrics to guide explainability research, underscoring the need for a broader reassessment of what precisely these metrics are intended to capture.

## Installation
To recreate the conda environment, run 
`conda env create -f environment.yml`.

To evaluate the sufficiency and comprehensiveness scores of the models, please see the rationale_benchmark/metrics.py in the [ERASER benchmark code](https://github.com/jayded/eraserbenchmark).

To see configuration details for glue_models/run_glue.py, please see the [original run_glue.py code](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) that we base our run_glue.py off of.


## Datasets
To download the datasets used in the paper, see instructions in the [ERASER benchmark code](https://github.com/jayded/eraserbenchmark)


## Citation
If you use our code, datasets, or concepts from our paper in your research, we would appreciate citing it in your work. Here is an example BibTeX entry for citing our paper:
```bibtex
@article{hsia2023goodhart,
  title={Goodhart's Law Applies to NLP's Explanation Benchmarks},
  author={Hsia, Jennifer and Pruthi, Danish and Singh, Aarti and Lipton, Zachary C},
  journal={arXiv preprint arXiv:2308.14272},
  year={2023}
}
```
## Contact
For any questions, feedback, or discussions regarding this project, please feel free to contact us:

- **Your Name**
  - Email: [jhsia2@cs.cmu.edu](mailto:jhsia2@cs.cmu.edu)
  - GitHub: [@jenhsia](https://github.com/jenhsia)

