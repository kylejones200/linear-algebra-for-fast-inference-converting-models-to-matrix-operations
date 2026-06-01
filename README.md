# Linear Algebra for Fast Inference

This project demonstrates using linear algebra for fast inference in machine learning and data science.

## Business context

Machine learning models often feel like black boxes. But once trained, many models --- especially in supervised learning --- can be expressed as nothing more than matrix multiplications and vector additions. This isn't just a theoretical convenience. It makes inference fast. Matrix operations are the backbone of modern computing hardware, optimized in GPUs, TPUs, and even low-power edge devices.

When we convert a trained model into a sequence of linear algebra steps, we can strip away unnecessary computation and reduce the entire prediction process to simple, fast operations. This is especially critical for serving models at scale, deploying to mobile or embedded environments, or optimizing inference pipelines.

Let's walk through some common models that can be rewritten as pure linear algebra. We also explain why this matters and how it helps us make predictions quickly in production environments.

## Article

Medium article: [Linear Algebra for Fast Inference](https://medium.com/@kylejones_47003/linearalgebrafastinference)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Linear algebra inference functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- System dimensions (n variables, m equations)
- Inference method (lstsq, pinv)
- Output settings

## Linear Algebra Methods

Inference methods:
- Least Squares (lstsq): Standard least squares solution
- Pseudoinverse (pinv): Moore-Penrose pseudoinverse
- Fast Computation: Optimized linear algebra operations

## Caveats

- By default, generates synthetic linear systems.
- Method selection depends on system properties.
- Large systems may require memory optimization.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).