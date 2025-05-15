# **Evaluation for 04-GEO-OMA24**

## **Task**

Your task is to create a **PyTorch-based deep learning pipeline** following best practices, making use of functionalities from the **torch** ecosystem, such as `torch`,`torchmetrics`, `torchvision`, `torchgeo`, and related libraries. The focus of this assignment is on **code structure, scalability, reusability, and deployability**.

A well-executed assignment should be designed in such a way that **you would have been able to use your own work before attending this class**.

### **What This Assignment is NOT About**

This assignment is **NOT** about:
- Training a perfect neural network  
- Designing a novel architecture  
- Writing a custom loss function from scratch  
- Reinventing fundamental concepts like stochastic gradient descent  

The primary objective is to demonstrate **stable and scalable deep learning pipelines**—this is the core focus of the course and the assignment.

However, if you choose to explore additional aspects—such as novel architectures or custom loss functions—I will gladly provide **constructive feedback** to help you improve your skills. That said, **the core requirements of the assignment must not suffer** due to excessive time spent on advanced explorations.

### **Default Task or Own Idea**

The default task is an **object detection** problem. You can use the **DOTA dataset** as a fallback if you do not want to find another object detection dataset. The DOTA dataset will be available on **Terrabyte**, partially preprocessed but **not yet deep-learning-ready** for PyTorch. Your job is to **prepare it accordingly** and integrate it into your pipeline.

However, if you prefer to work on a different task, you are free to do so, provided that:
- It is more complex than **basic image classification** (as that would be too easy).
- It requires a structured deep learning pipeline, as discussed during class.

---

## **Evaluation Scheme**

Each assignment will be evaluated based on the following criteria. The scheme is **task-agnostic**, focusing on best practices, scalability, reusability, and project/code structure.

✅ Criteria Met

🟦 Criteria Partly Met

❌ Criteria Not Met

📝 Comments

⭐ Excellent Performance


| Points   | Grade                 | Student's Grade |
| ---------| --------------------- |-----------------|
| 95 - 100 | 1,0                   |     100 🎓      |
| 90 - 94  | 1,3                   |                 |
| 85 - 89  | 1,7                   |                 |
| 80 - 84  | 2,0                   |                 |
| 75 - 79  | 2,3                   |                 |
| 70 - 74  | 2,7                   |                 |
| 65 - 69  | 3,0                   |                 |
| 60 - 64  | 3,3                   |                 |
| 55 - 59  | 3,7                   |                 |
| 50 - 54  | 4,0                   |                 |
| 0 - 49   | 5,0 (failed)          |                 |


### Total: 100/100 points:

📝 Comments: see also comments in the code.

#### **1. Code Quality & Structure (25/25 points)**
- **✅Modularity (5/5 points):**  
  Clear separation of concerns (e.g., data processing, model definition, training loop, evaluation, model export/import, and inference).

  📝 your structure is well setup and intuitive to follow. The single scripts are not too bloated.
- **✅Documentation (5/5 points):**  
  A `README.md` file with sections for both **users** (inference) and **developers** (training & validation), including:
  - Environment setup (platform-agnostic).
  - Instructions for running inference.
  - Instructions for training and validation.
- **✅Concluding Discussion (5/5 points):**  
  Provide a well-structured discussion analyzing:
  - Model performance.
  - Potential failure cases.
  - Possible improvements.
- **✅Efficiency & Scalability (5/5 points):**  
  Code should support batching, GPU acceleration, and efficient data loading.
- **✅Proper Use of PyTorch Best Practices (5/5 points):**  
  Utilization of `torch.nn`, `torch.utils.data`, `torch.optim`, etc., without unnecessary custom implementations.

#### **2. Data Handling & Preprocessing (20/20 points)**
- **✅Dataset Handling (10/10 points):**  
  Efficient data loading using a **PyTorch-based `Dataset` and `DataLoader`**.
- **✅Handling of Edge Cases (5/5 points):**  
  The pipeline should handle missing/corrupt data, apply correct normalization, and implement appropriate dataset splitting.

📝⭐ Due to your extensive and very unique data preparation pipeline, I see this task and the next in general as more than sufficiently worked on.
- **✅Data Augmentation & Preprocessing (5/5 points):**  
  Use `torchvision.transforms.v2` (or equivalent) for augmentation in a reusable manner instead of writing custom implementations.

#### **3. Model Design & Implementation (10/10 points)**
- **✅Appropriate Architecture (5/5 points):**  
  Use a suitable model architecture for the given task (e.g., CNNs, ResNet, ViTs).

📝 What I liked is that reading your implementation was so clear due to its good structure, that I  might have found an issue and provided some thoughts about it and another implementation, check the comments in the `README.md`. 
- **✅Use of Pretrained Models / Transfer Learning (5/5 points):**  
  If applicable, leverage `torchvision.models` or `torchgeo.models` correctly.

#### **4. Training Pipeline & Optimization (20/20 points)**
- **✅Loss Function & Optimization (5/5 points):**  
  Use an appropriate loss function and optimizer (e.g., `CrossEntropyLoss`, `Adam`, `SGD`).

📝⭐ When I drafted the course, I did not think about that someone implements a custom loss function, amazing, even when it is an established one. I hope you learned a lot from it. 
- **✅Training Loop Implementation (5/5 points):**  
  The training loop should include proper batching, gradient computation, and GPU acceleration.
  
📝⭐ The usage of slurm on terrabyte is of course the best way, well done.
- **✅Logging & Monitoring (10/10 points):**  
  Use **TensorBoard** for tracking loss, performance metrics, learning rate, etc.
  - **Save the TensorBoard event files on Terrabyte** and document their location in the `README.md`.

#### **5. Evaluation & Results (15/15 points)**
- **✅Model Evaluation Metrics (10/10 points):**  
  Proper computation of **accuracy, precision, recall, F1-score**, or **mAP** (for object detection).
- **✅Generalization & Robustness (5/5 points):**  
  Evaluate the model on an **unseen test split** and discuss potential failure cases.

#### **6. Reproducibility & Deployment Readiness (10/10 points)**
- **✅Reproducibility (5/5 points):**  
  The code should run without errors on another machine. **Dedicated sections** in the code for adapting paths/settings to different machines are allowed and encouraged for this beginners example.
- **✅Deployment Readiness (5/5 points):**  
  The model should be **exportable to `.pth` (checkpoint format)**, with a dedicated **inference script and/or notebook** for model loading and inference.

  # 📝 Overall Comments

I am not sure how many Deep Learning projects you have already done, but this is a solid foundation. I encourage you to keep working in the field, the assignments clearly shows you understand the fundamentals and are able to implement them to solve a more or less custom task.
One thing which I strongly encourage you to look into is to write cleaner/better code. It is not that your code is unreadable or bad, but I think cleaner code will help you to develop your skills in Deep Leaning much faster. How to do that? Simply code as much as you can and use `ruff` while doing so https://docs.astral.sh/ruff/ I have already configured it in the `pyproject.toml` file, so you can use it right away. Just install `ruff` and the VSCode extension and you are good to go.

