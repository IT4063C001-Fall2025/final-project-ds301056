# Fast Food & General Nutrition Analysis

<!-- by Derek Singleton -->

<!-- Edit the title above with your project title -->

## Project Overview

Topic

The project explores the nutritional value of fast food compared with general food items. The goal is to identify patterns in calories, fats, proteins, and sugar content across food categories and brands, then compare those findings with authoritative USDA nutritional data.

This topic is important because dietary choices directly impact health outcomes, and fast food consumption is widespread. By highlighting nutrition differences and trends, the project may help consumers make better-informed food decisions.

Project Questions

1. How does the nutritional profile (calories, fat, protein, sugar) of popular fast food items compare to general grocery store food items?

2. Which fast food chains provide healthier menu options on average?

3. Are certain cuisines or food categories (burgers, salads, beverages, desserts) consistently higher or lower in key nutrients?

4. How well do Kaggle datasets align with USDA’s official nutrition data?
5.

What Would an Answer Look Like?

Bar chart: Average calories per serving by fast food chain.

Boxplot: Distribution of sugar content across food categories.

Heatmap: Nutrient density (calories, protein, fat, sugar) by food type.

Scatter plot: Calories vs protein content for fast food vs general food items.

Data Sources

Kaggle Dataset (CSV) — Fast Food Nutrition Dataset

https://www.kaggle.com/datasets/ulrikthygepedersen/fastfood-nutrition

Contains nutrition facts for menu items from popular fast food restaurants (calories, fat, protein, etc.)

Kaggle Dataset (CSV) — Nutritional Content of Food

https://www.kaggle.com/datasets/thedevastator/the-nutritional-content-of-food-a-comprehensive

Comprehensive dataset of nutrition facts for thousands of foods from various categories, not limited to fast food.

USDA FoodData Central API — FoodData Central

https://fdc.nal.usda.gov/

Authoritative API that provides verified nutrition data from the USDA.
Example endpoint:

## Self Assessment and Reflection

<!-- Edit the following section with your self assessment and reflection -->

it took me a bit to find relatable datasets and to format the .pynb. after some trial and error i got it. I forgot you could change a cell to be type markdown. - at first it thought i just had a bunch of syntax errors and it wanted the answers to be in comments.

### Self Assessment

<!-- Replace the (...) with your score -->

| Category          | Score   |
| ----------------- | ------- |
| **Setup**         | 10 / 10 |
| **Execution**     | 20 / 20 |
| **Documentation** | 10 / 10 |
| **Presentation**  | 30 / 30 |
| **Total**         | 70 / 70 |

### Reflection

<!-- Edit the following section with your reflection -->

#### What went well?

(The data exploration and cleaning)

#### What did not go well?

the graphs can be finniky to format

#### What did you learn?

the IQR can be one of the best ways to find outliers q3 - q1 -= (lower and upper bounds)

#### What would you do differently next time?

I would probably pull the api data from the gov api and use it as the standard and weight to see if the values in the dataset are trustable

---

## Getting Started

### Installing Dependencies

To ensure that you have all the dependencies installed, and that we can have a reproducible environment, we will be using `pipenv` to manage our dependencies. `pipenv` is a tool that allows us to create a virtual environment for our project, and install all the dependencies we need for our project. This ensures that we can have a reproducible environment, and that we can all run the same code.

```bash
pipenv install
```

This sets up a virtual environment for our project, and installs the following dependencies:

- `ipykernel`
- `jupyter`
- `notebook`
- `black`
  Throughout your analysis and development, you will need to install additional packages. You can can install any package you need using `pipenv install <package-name>`. For example, if you need to install `numpy`, you can do so by running:

```bash
pipenv install numpy
```

This will update update the `Pipfile` and `Pipfile.lock` files, and install the package in your virtual environment.

## Helpful Resources:

- [Markdown Syntax Cheatsheet](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [Dataset options](https://it4063c.github.io/guides/datasets)
