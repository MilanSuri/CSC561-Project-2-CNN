# CSC561-Project-2-NN

This is a fully functional take on the CSC561 gradients project. We implement a custom symbolic and functional calculus and vector operations, and then use those operations to build ReLU, BCE, Gradient Descent, Sigmoids, and eventually train a logistic regression model.

___
## Structure:

In the `cMath` folder, we have our custom math implementations and tests for calculus operations and vector operations.
These form the backbone of the models.

Then in the `src` file, there is the pipeline where the models are developed. So you have optimizers, activation, functions, and loss functions. These all are used for making the ML model learn from the other points.

___
## Contributing:

- Clone the repository using `git clone`
- Ensure you're on the latest version: `git pull`
- Create a new branch to submit a PR
- Once you're ready to push your branch/submit a PR run: `git commit -m {commit msg}` and then `git push -u origin {branch_name}`
- Then write a detailed summary of what your proposed addition does.

### Running:
- You can install UV (dependency manager) through:
  - macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
  - Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh | iex"
  - Homebrew: brew install uv
  - Pip: pip install uv
- We use UV for this project so you can install UV and run the command: uv sync.
- To run the full pipeline run the `main.py` file.

___
## Interpretability:

I utilized the SHAP library in python to generate an explainer for how the different inputs in the breastcancer dataset impact the prediction.

This is seen in the photo: `shap_summary.png` in `src` folder.

![Shap Explainer](src/shap_summary.png)

This is the Shap explainer. At 0.0 you have no effect on the model, each point shows how the model interpreted a patient. 
The colors correspond with the size of the change, and then the +/- show how it impacts the prediction. The classification of 0 meant benign and 1 is malignant. Therefore if it's a negative impact, it's more associated with being benign, and if it's positive more malignant.
