from haversine import haversine, Unit
from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# We want to give coordinates
# And find the radius of curvature amplitude and wave length


# CSV reader
# data = pd.read_csv('river_data.csv')
# easting = data["Easting"].str.split().str[0].astype(float)
# northing = data["Northing"].str.split().str[0].astype(float)

# river_utm_data = np.column_stack((easting, northing))


# Take UTM data, and find their distance from the start of a curve
def process_data(start, UTM_data):
    num_data_points = len(UTM_data)
    numerical = np.zeros((num_data_points, 2))
    for i in range(1,num_data_points):
        numerical[i] = UTM_data[i] - start
    return numerical.transpose()

# Interpolate points to fit a line to the data

# UTM_data = [Easting, Northing]
def get_curve(start,UTM_data,smooth_factor):
    clean_data = process_data(start, UTM_data)
    b_spline, u = interpolate.make_splprep(clean_data, s=smooth_factor)
    return b_spline


def plot_curve(curve, UTM_data):
    u = np.linspace(0,1,500)
    curve_eval = curve(u)
    x_spline = curve_eval[0]
    y_spline = curve_eval[1]

    # Plotting code for testing code -- gpt helped
    # plt.scatter(x_spline, y_spline, label="Measured points")
    plt.plot(x_spline, y_spline, label="B-spline")
    plt.scatter(UTM_data[0], UTM_data[1], label = "Data", s=3, color = "red")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.axis("equal")
    plt.legend()
    plt.show()

def find_radius_of_curvature(param_curve,x,y):
    # # clean_UTM_data is 2 x num_data_points. Take the transpose to iterate easily by data point.
    # data = clean_UTM_data.transpose()
    # for i in range(1, len(data) - 1):
    
    u = get_param(x,y,param_curve)
    deriv_1 = param_curve.derivative(1)
    deriv_2 = param_curve.derivative(2)


    kappa = (deriv_1(u)[0]*deriv_2(u)[1] - deriv_1(u)[1]*deriv_2(u)[0]) /(((deriv_1(u)[0])**2 + (deriv_1(u)[1])**2)**(3/2))
    return 1/abs(kappa)
        
    
#  Pass the parametric B-spline and target easting and northing values (x_data, y_data). This function gets the corresponding parameter from easting and northing values
# Precision is how many u values you want to test. Default = 1000
def get_param(x_data,y_data, param_curve, precision = 1000):

    u_lin = np.linspace(0,1, precision)
    # Find the parameter u that produces x_data and y_data
    # minimizing the 2-norm
    diff = sys.maxsize
    u_closest=0
    for u in u_lin:
        x_u, y_u = param_curve(u)
        diff_test = np.linalg.norm([x_data - x_u, y_data - y_u],2) #Minimize the two norm
        if diff_test < diff:
            diff = diff_test
            u_closest = u
    return u_closest




def get_widths(left_bank_UTM, right_bank_UTM):

    # Get the parametric curves
    left_curve = get_curve(left_bank_UTM)
    right_curve = get_curve(right_bank_UTM)

    # Get tangent lines
    left_curve_deriv1 = left_curve.derivative(1)
    right_curve_deriv1 = right_curve.derivative(1)

    # Sample u values, third argument is number of places to calculate the width
    u = np.linspace(0,1,500)
    return

# CSV reader to read in Jack and Tian's data.
data = pd.read_csv('Path1_UTM_tian.csv', header=None)
easting = data[0]
northing = data[1]

river_utm_data = np.column_stack((easting, northing))

# param_curve=get_curve(river_utm_data[0], river_utm_data)
# plot_curve(param_curve)

processed_data = process_data(river_utm_data[0], river_utm_data)
print(processed_data.transpose()[6])

# Find Radius of curvature:
s_vals = [0,10,30,100,300,500,1000,2000,2500,3000, 4000]
for s in s_vals:
    param_curve = get_curve(river_utm_data[0], river_utm_data,s)
    radius_of_curvature = find_radius_of_curvature(param_curve, -3.97578385,  0.15121202)
    
    print("Radius of curvature is ", radius_of_curvature, "for s =", s)
    if s == 500:
            plot_curve(param_curve,processed_data)