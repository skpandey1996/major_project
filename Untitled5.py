
# coding: utf-8

# In[8]:


from pandas import read_csv
import tkinter as tk 
from pandas import DataFrame
from sklearn import linear_model
from sklearn import metrics  
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
  
Crop_yield = read_csv(r'C:\Users\subham\Desktop\main_assam9.csv')
 
df = DataFrame(Crop_yield,columns=['area','yield'])
 
plt.scatter(df['area'], df['yield'], color='red')
plt.title('area Vs yield', fontsize=14)
plt.xlabel('Area', fontsize=14)
plt.ylabel('Yield', fontsize=14)
plt.grid(True)
plt.show()
 

X = df[['area']]
Y = df['yield']
 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)  
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
Y_predictions = regr.predict(X_test)
dff = DataFrame({'Actual': y_test, 'Predicted': Y_predictions})  
print(dff) 
print (Y_predictions)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
New_Area = 2.75
print ('Predicted yield: \n', regr.predict([[New_Area]]))
#regr.score(X_train,y_train)
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_predictions)))

errors = abs(Y_predictions - y_test)

# Print out the mean absolute error (mae)
#print('Mean Absolute Error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# tkinter GUI
root= tk.Tk() 
 
canvas1 = tk.Canvas(root, width = 1200, height = 450)
canvas1.pack()


# with sklearn
Intercept_result = ('Intercept: ', regr.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
canvas1.create_window(260, 220, window=label_Intercept)

# with sklearn
Coefficients_result  = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
canvas1.create_window(260, 240, window=label_Coefficients)


# New_Interest_Rate label and input box
label1 = tk.Label(root, text='Type Total Area: ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# New_Unemployment_Rate label and input box
#label2 = tk.Label(root, text=' Type Average Annual Rainfall: ')
#canvas1.create_window(120, 120, window=label2)

#entry2 = tk.Entry (root) # create 2nd entry box
#canvas1.create_window(270, 120, window=entry2)

# New_Interest_Rate label and input box
#label3 = tk.Label(root, text='Type Average Annual Temperature: ')
#canvas1.create_window(100, 100, window=label1)

#entry3 = tk.Entry (root) # create 3rd entry box
#canvas1.create_window(270, 100, window=entry1)



def values(): 
    global Area #our 1st input variable
    Area = float(entry1.get()) 
    
    #global New_Unemployment_Rate #our 2nd input variable
    #New_Unemployment_Rate = float(entry2.get()) 
    
    Prediction_result  = ('Predicted Annual Crop Yield: ', regr.predict([[Area]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='blue')
    canvas1.create_window(260, 280, window=label_Prediction)
    
button1 = tk.Button (root, text='Predict Annual Crop Yield',command=values, bg='blue') # button to call the 'values' command above 
canvas1.create_window(270, 150, window=button1)

root.mainloop()

