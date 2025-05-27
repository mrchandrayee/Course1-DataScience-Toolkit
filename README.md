# Course 1: Data Science Toolkit

A comprehensive collection of data science learning materials, tutorials, and practical implementations covering fundamental to advanced topics in data science, machine learning, and deep learning.

## 🚀 Project Overview

This repository serves as a complete toolkit for learning data science, featuring hands-on tutorials, practical implementations, and real-world examples. The content is organized into structured learning modules that progress from basic Python programming to advanced neural networks and machine learning algorithms.

## 📚 Repository Structure

### Core Learning Modules (`DS_New/`)

#### **Fundamentals**
- **02 Python Intro.ipynb** - Introduction to Python programming for data science
- **03 Gradient Descent.ipynb** - Implementation and visualization of gradient descent algorithm

#### **Regression & Valuation**
- **04 Multivariable Regression.ipynb** - Multi-variable regression analysis techniques
- **04 Valuation Tool.ipynb** - Real estate valuation using machine learning
- **boston_valuation.py** - Boston housing price prediction tool

#### **Classification & Bayesian Methods**
- **06 Bayes Classifier - Pre-Processing.ipynb** - Data preprocessing for Bayes classification
- **07 Bayes Classifier - Training.ipynb** - Training Bayesian classifiers
- **07 Bayes Classifier - Testing, Inference & Evaluation.ipynb** - Model evaluation and testing
- **08 Naive Bayes with scikit-learn.ipynb** - Implementing Naive Bayes using scikit-learn

#### **Deep Learning & Neural Networks**
- **09 Neural Nets Pretrained Image Classification.ipynb** - Using pretrained models for image classification
- **10 Neural Nets - Keras CIFAR10 Classification.ipynb** - CIFAR-10 image classification with Keras
- **11 Neural Networks - TF Handwriting Recognition.ipynb** - Handwritten digit recognition using TensorFlow

### Additional Resources (`Python Rahul Sir CTC/`)
- Advanced Python programming concepts
- Pandas data manipulation tutorials
- Statistics fundamentals
- Linear and logistic regression implementations
- Tree-based modeling approaches

### External Libraries & Frameworks
- **PyTorch Deep Learning** - Complete PyTorch tutorials and implementations
- **TensorFlow Deep Learning** - TensorFlow examples and best practices
- **Zero to Mastery ML** - Comprehensive machine learning course materials

## 🛠️ Technologies & Libraries

- **Python** - Primary programming language
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Machine learning algorithms
- **TensorFlow/Keras** - Deep learning framework
- **PyTorch** - Deep learning and neural networks
- **NLTK** - Natural language processing
- **BeautifulSoup** - Web scraping and HTML parsing

## 🎯 Learning Objectives

By working through this toolkit, you will learn:

1. **Python Fundamentals** for data science
2. **Data Preprocessing** and cleaning techniques
3. **Statistical Analysis** and hypothesis testing
4. **Machine Learning Algorithms** (supervised and unsupervised)
5. **Deep Learning** with neural networks
6. **Computer Vision** applications
7. **Natural Language Processing** basics
8. **Model Evaluation** and validation techniques

## 🚦 Getting Started

### Prerequisites
```bash
Python 3.7+
Jupyter Notebook or JupyterLab
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/mrchandrayee/Course1-DataScience-Toolkit.git
cd Course1-DataScience-Toolkit
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras torch nltk beautifulsoup4 wordcloud pillow
```

3. Start Jupyter Notebook:
```bash
jupyter notebook
```

## 📖 Usage Examples

### Boston Housing Price Prediction
```python
from DS_New.boston_valuation import get_log_estimate
import numpy as np

# Predict house price
log_price = get_log_estimate(nr_rooms=6, students_per_classroom=20, next_to_river=True)
actual_price = np.exp(log_price) * SCALE_FACTOR
print(f"Estimated house price: ${actual_price:,.2f}")
```

### Image Classification with Neural Networks
Navigate to `09 Neural Nets Pretrained Image Classification.ipynb` for complete image classification workflows.

## 📊 Datasets Included

- **MNIST** - Handwritten digit recognition
- **CIFAR-10** - Object recognition in images
- **Boston Housing** - Real estate price prediction
- **Spam Data** - Email spam classification
- **Custom datasets** for various classification tasks

## 🔧 Project Features

- ✅ Beginner-friendly tutorials with step-by-step explanations
- ✅ Real-world dataset implementations
- ✅ Complete machine learning pipelines
- ✅ Deep learning model architectures
- ✅ Data visualization examples
- ✅ Model evaluation metrics and techniques
- ✅ Production-ready code examples

## 📁 Note about Large Files

Some large files have been excluded from this repository due to GitHub's 100MB file size limit:

- Archive files (*.zip, *.tar.gz, *.rar, *.7z)
- Large dataset files (>100MB CSV files)
- Specific excluded items:
  - zero-to-mastery-ml-master.zip (240.37 MB)
  - tensorflow-deep-learning-main.zip (208.89 MB) 
  - pytorch-deep-learning-main.zip (411.58 MB)

For large files, consider using [Git LFS (Large File Storage)](https://git-lfs.github.com/) if needed.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

- **Author**: MR Chandrayee
- **Repository**: [Course1-DataScience-Toolkit](https://github.com/mrchandrayee/Course1-DataScience-Toolkit)

## 🌟 Acknowledgments

- Thanks to all the open-source contributors who made the libraries used in this project possible
- Special recognition to the data science community for sharing knowledge and best practices

---

⭐ **Star this repository if you find it helpful!** ⭐
