### Overview- Energy Generation a Global Perspective

This project uses data from the World Data Institute to predict Energy Generation classes (very low, low, mid, high) for 35000 power plants located around the world. This dataset utilizes various factors, like fuel type, capacity, energy generation data from 2013-2019, and location. These variables are important for determined which class the powerplant 

Deployment Link:
Deployment Github:
Tableau Public Link:

# Dataset Description

The dataset includes several variables pertaining the the plants. Capcaity, generation data, location, year built, data source were all included. 

# Business Problem 

The project is comissioned by the Internation Panel of Climate Change to gain a better understanding of global energy generation. This data will be used to make policy decisions as we transition to a greener future. 

What impact will it have if we can perdict the generation class of power plants?
What are some of the leading features that lead to this classification?
What is the global distribution of power plants and type, and how can this data be used for future policy decisions?

# Goals

Predictive Analysis: To predict generation class (very low, low, mid, high) of different power plants across the world.. 
Indicator Analysis: To identify key features that contirbute to this classification. 

# Key Questions

1. What preprocessing steps do we need to take to create an effective predictive model?
2. Which type of predictive model gets the best results?
3. What hypertuning techniques are used to tune the model?
4. What are the benefits of our model?

# Target Variable Investigation

In the target variable, average generation, we started with four classes:
very low
low 
mid
high

And tested different models to determine the best results. 
The notebook contains the following types of content:

## Preprocessing Steps...
Converted commission year to plant age
Removed plants with multiple fuel types because distribution is not known, focsuing on the majority
Grouping renewable fuels
Calculating Average Generation from actual, using estimated if not present
Inpute missing plant ages by mean
Recalculate average generation and binning into generation classes

## Exploratory data analysis
# Explored Descriptive statistics
![finalimage1]("[https://github.com/silver032/Energy-Generation-Classifier-phase-5-project-/blob/main/images/finalimage1.png"])
# Descriptive statistics


# Displaying Confusion Matrix for best tuned model...

# Displaying Feature Importance For Best model...


## Conclusions, next steps...
Effectiveness of the Random Forest Classifier: 94.6 - Highly Effective

Significance of Feature Importance: Capacity and Primary fuel 3. 
The Value of Data Augmentation and Scalability: more ro...
**Code Cell**: ...


