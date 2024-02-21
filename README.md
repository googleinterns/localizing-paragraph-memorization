# Localizing and Controlling Paragraph Level Recitation

Explores how small open source LMs like GPT-Neo implement paragraph level recitation from the training data. Includes helper scripts and exploratory Jupyter notebooks. Work done by [Niklas Stoehr](https://niklas-stoehr.com/) during his winter 2023 research internship.

**Not an official google project**
## Project Structure

### utils
__________________________________________________
helper scripts with basic functionality that is used in different notebooks

- patching
- evaluation
- dataLoaders
- gradient
- intervening
- localizing
- modelHandlers


### notebooks
__________________________________________________
notebooks to reproduce the main experiments

#### 1 descriptive
        - explorative
        - token pertubation 
#### 2 localizing
        - activation patching
        - gradient-based attribution
                - parameter gradients
                - activation gradients
        - attention head analysis
#### 3 editing
    
    
## paragraphs
__________________________________________________
CSV file of some paragraphs that are memorized  by GPT-neo-125M
