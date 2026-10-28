from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
from scipy.optimize import root
from scipy.signal import find_peaks

# We want to give coordinates
# And find the radius of curvature amplitude and wave length


# CSV reader
# data = pd.read_csv('river_data.csv')
# easting = data["Easting"].str.split().str[0].astype(float)
# northing = data["Northing"].str.split().str[0].astype(float)

# river_utm_data = np.column_stack((easting, northing))


###################
# DATA PROCESSING AND CURVE FINDING
###################
# Take UTM data, and find their distance from the start of a curve

# TODO: Test, see what I want to do with multiple curves
def process_data(UTM_data, UTM_data_2 = None):
    if(UTM_data_2 is None):
        shared_origin = [0,0]
        easting = UTM_data[0]
        northing = UTM_data[1]

        data = np.column_stack((easting, northing))
        # AI helped me get unique entries in a matrix
        _, idx = np.unique(data, axis=0, return_index=True)
        data = data[np.sort(idx)]
        numerical = np.zeros(np.shape(data))
        for i in range(len(data)):
            numerical[i] = data[i] - shared_origin
        return numerical.transpose()
    else:
        easting1 = UTM_data[0]
        northing1 = UTM_data[1]
        easting2 = UTM_data_2[0]
        northing2 = UTM_data_2[1]

        data1 = np.column_stack((easting1, northing1))
        _, idx = np.unique(data1, axis=0, return_index=True)
        data1 = data1[np.sort(idx)]

        data2 = np.column_stack((easting2, northing2))
        _, idx2 = np.unique(data2, axis=0, return_index=True)
        data2 = data2[np.sort(idx2)]

        shared_origin = (data1[0] + data2[0])/2
        numerical1 = np.zeros(np.shape(data1))
        numerical2 = np.zeros(np.shape(data2))
        for i in range(len(data1)):
            numerical1[i] = data1[i] - shared_origin
        for i in range(len(data2)):
            numerical2[i] = data2[i] - shared_origin
        return [numerical1.transpose(), numerical2.transpose()]
        

# Interpolate points to fit a line to the data

# UTM_data = [Easting, Northing]
def get_curve(UTM_data, UTM_Data2 = None, smooth_factor=1000):
    if(UTM_Data2 is None):
        clean_data = process_data(UTM_data)
        b_spline, u = interpolate.make_splprep(clean_data, s=smooth_factor)
        return b_spline
    else:
        clean_data = process_data(UTM_data, UTM_Data2)
        b_spline1, u = interpolate.make_splprep(clean_data[0], s=smooth_factor)
        b_spline2, t = interpolate.make_splprep(clean_data[1], s=smooth_factor)
        return [b_spline1, b_spline2]

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

def find_radius_of_curvature(param_curve,x,y):
    # # clean_UTM_data is 2 x num_data_points. Take the transpose to iterate easily by data point.
    # data = clean_UTM_data.transpose()
    # for i in range(1, len(data) - 1):
    
    u = get_param(x,y,param_curve)
    deriv_1 = param_curve.derivative(1)
    deriv_2 = param_curve.derivative(2)


    kappa = (deriv_1(u)[0]*deriv_2(u)[1] - deriv_1(u)[1]*deriv_2(u)[0]) /(((deriv_1(u)[0])**2 + (deriv_1(u)[1])**2)**(3/2))
    return 1/abs(kappa)
        

##############################
# FIND NORM LINE(S) AND WIDTHS
##############################

def get_norm_lines(curve, num_lines = 20):
   
    norm_lines = []

    right_curve_deriv1 = curve.derivative(1)

    u = np.linspace(0,1, num_lines)
    norm_lines = []
    for num in u:
        deriv = right_curve_deriv1(num)
        normal_slope = [-deriv[1], deriv[0]]
        normal_line = find_norm_line(normal_slope, curve, num)
        norm_lines.append(normal_line)
    return norm_lines


# Get normal lines from one curve calling the function get_norm_lines()
# Get the width -- distance from the start of this line (2-norm) to the other edge.
def get_widths(other_curve, norm_lines):

    widths = np.zeros(len(norm_lines))
    # Sample u values, third argument is number of places to calculate the width. 
    # the parameter, u, ranges from 0, to 1. Here we choose 500 sample points for the width
    i = 0 
    for line in norm_lines:
        print(line(find_first_intersect(other_curve, line, 10**4)))
        
        width = np.linalg.norm(np.array(line(find_first_intersect(other_curve, line, 10**4))) - np.array(line(0)))
        widths[i] = width
        i+=1
    # From here, we need to find the normal curve to the tangent line on one bank and intersect it with the other shoreline.
    return widths



# Returns the parametric normal curve
def find_norm_line(norm_slope, param_curve, u):

    m_1 = norm_slope[0]
    m_2 = norm_slope[1]
    x = param_curve(u)[0]
    y = param_curve(u)[1]

    normal_curve = lambda param: [m_1 * param + x, m_2*param + y]
    return normal_curve



##############################
# NORM LINE INTERSECTION
##############################

def close_point_to_line(point, line):
    orig_point = np.array(line(0))
    new_point = np.array(line(1))
    direction_vec_line = new_point - orig_point
    vec_to_point = point - orig_point
    t = np.dot(direction_vec_line, vec_to_point) / np.dot(direction_vec_line, direction_vec_line)
    return t


def find_first_intersect(other_curve, norm_line, num_params= 100000):
    # We make a guess that the intersection is about 20 meters away
    guess_dist = 6 

    # Based on guess distance, find the corresponding point in x, y
    theta = math.atan2((norm_line(1)[1]- norm_line(0)[1]),(norm_line(1)[0] - norm_line(0)[0]))
    d_y = guess_dist* math.sin(theta)
    d_x = guess_dist * math.cos(theta)

    guess_point = np.array(norm_line(0)) + [d_x, d_y]

    # We make our guess parameter guess, by minizing the parameter on each line closest to this point
    guess_curve_param = get_param(guess_point[0], guess_point[1], other_curve)
    guess_line_param = get_param(guess_point[0], guess_point[1], norm_line)
    print("curve param guess is ", guess_curve_param)
    print("line param guess is ", guess_line_param)

    def difference(params):
        u,t = params
        return np.array(other_curve(u)) - np.array(norm_line(t))
    

    solution = root(difference, [guess_curve_param, guess_line_param])
    u, t = solution.x
    return t

def get_width_from_data_point(easting, northing):
    return

##############################
# PLOTTING
##############################


def plot_curve(curve, UTM_data = None):
    u = np.linspace(0,1,500)
    curve_eval = curve(u)
    x_spline = curve_eval[0]
    y_spline = curve_eval[1]
    # Plotting code for testing code -- gpt helped
    # plt.scatter(x_spline, y_spline, label="Measured points")
    plt.plot(x_spline, y_spline, label="B-spline")
    if not UTM_data == None:
        plt.scatter(UTM_data[0], UTM_data[1], label = "Data", s=3, color = "red")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.axis("equal")
    plt.legend()

def plot_width(norm_line, norm_line_width):
    # find angle of river 
    theta = math.atan2((norm_line(1)[1]- norm_line(0)[1]),(norm_line(1)[0] - norm_line(0)[0]))
    d_y = norm_line_width * math.sin(theta)
    d_x = norm_line_width * math.cos(theta)

    end = np.array(norm_line(0)) + [d_x, d_y]

    final_t = close_point_to_line(end, norm_line)
    t = np.linspace(0, final_t, 1000)
    plt.plot(norm_line(t)[0], norm_line(t)[1], label=str(round(norm_line_width,2)))

def plot_curves_and_widths(widths, norm_lines, left_curve, right_curve):
    # norm_lines is list of all norm lines
    plot_curve(left_curve)
    plot_curve(right_curve)
    for i in range(len(norm_lines)):
        t = find_first_intersect(left_curve, norm_lines[i], 10**4)
        print("parameter for line intersection is", t)
        pt = norm_lines[i](t)
        plt.plot(*pt, 'ko', markersize = 4)
        plot_width(norm_lines[i], widths[i])
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.axis("equal")
    plt.legend()
    plt.show()

        


# CSV reader to read in Jack and Tian's data.
data = pd.read_csv('Path2_UTM_tian.csv', header=None)

param_curve=get_curve(data)
# plt.show()

# Cehck how much smoothing is best for radius of curvature
# def radius_of_curvature_check():
#     s_vals = [0,10,30,100,300,500,1000,2000,2500,3000, 4000]
#     for s in s_vals:
        # param_curve = get_curve(river_utm_data,s)
        # radius_of_curvature = find_radius_of_curvature(param_curve, -3.97578385,  0.15121202)
        
        # print("Radius of curvature is ", radius_of_curvature, "for s =", s)
        # if s == 500:
        #         plot_curve(param_curve,processed_data)


# process data -- subtract the start utm value from each data point and take the transpose (transpose is necessary for the spline)
# processed_data = process_data(river_utm_data[0], river_utm_data)
# print(processed_data.transpose()[6])


def get_wavelengths(param_curve, dist=20):
    u = np.linspace(0,1,500)
    x_centered_curve = param_curve(u)[0] - np.mean(param_curve(u)[0])
    y_centered_curve = param_curve(u)[1] - np.mean(param_curve(u)[1])
    curve_centerline = np.polynomial.Polynomial.fit(x_centered_curve, y_centered_curve, 1)
    # plt.plot(x_centered_curve, y_centered_curve, label='Centered Curve')
    # plt.plot(x_centered_curve, curve_centerline(x_centered_curve), label='Fitted Centerline', color='orange')
    # plt.xlabel("Easting (m)")
    # plt.ylabel("Northing (m)")
    # plt.axis("equal")
    # plt.show()
    theta = np.arctan(curve_centerline(1))
    print(theta)
    param_curve_rotated = [param_curve(u)[0]*np.cos(theta) + param_curve(u)[1]*np.sin(theta), -param_curve(u)[0]*np.sin(theta) + param_curve(u)[1]*np.cos(theta)]

    peaks, _ = find_peaks(param_curve_rotated[1], distance=dist)
    # Calculate wavelengths as distances between consecutive extrema
    wavelengths = []
    x_curve = param_curve_rotated[0]
    x_peaks = x_curve[peaks]
    plt.plot(param_curve_rotated[0], param_curve_rotated[1], label='Rotated Curve')
    plt.scatter(param_curve_rotated[0][peaks], param_curve_rotated[1][peaks], color='red', label='Extrema')
    plt.axis("equal")
    plt.show()
    for i in range(len(x_peaks) - 1):
        wavelength = x_peaks[i + 1] - x_peaks[i]
        wavelengths.append(wavelength)
    return wavelengths, x_peaks

# wavelengths, x_peaks = get_wavelengths(param_curve)
# print("Shape of peaks indices:", np.shape(x_peaks))
# print(wavelengths)
# np.mean(wavelengths)
# print("Mean wavelength is ", np.mean(wavelengths))
# plot_curve(param_curve)
# plt.scatter(param_curve(x_peaks)[0], param_curve(x_peaks)[1], color='red', label='Extrema')
# plt.show()

def find_center_of_curvature(param_curve, u):
    deriv_1 = param_curve.derivative(1)
    deriv_2 = param_curve.derivative(2)

    kappa = (deriv_1(u)[0]*deriv_2(u)[1] - deriv_1(u)[1]*deriv_2(u)[0]) /(((deriv_1(u)[0])**2 + (deriv_1(u)[1])**2)**(3/2))
    radius_of_curvature = 1/abs(kappa)

    # Calculate the unit normal vector
    normal_vector = np.array([-deriv_1(u)[1], deriv_1(u)[0]])
    normal_vector /= np.linalg.norm(normal_vector)

    # calculate values for a point slightly offset from the original point to find the way the curve is pointing
    u_offset = u + 1e-6
    kappa_offset = (deriv_1(u_offset)[0]*deriv_2(u_offset)[1] - deriv_1(u_offset)[1]*deriv_2(u_offset)[0]) /(((deriv_1(u_offset)[0])**2 + (deriv_1(u_offset)[1])**2)**(3/2))
    radius_of_curvature_offset = 1/abs(kappa_offset)

    # Calculate the unit normal vector
    normal_vector_offset = np.array([-deriv_1(u_offset)[1], deriv_1(u_offset)[0]])
    normal_vector_offset /= np.linalg.norm(normal_vector_offset)

    center_offset_pos = np.array(param_curve(u_offset)) + radius_of_curvature_offset * normal_vector_offset
    center_offset_neg = np.array(param_curve(u_offset)) - radius_of_curvature_offset * normal_vector_offset
        
    # Calculate the center of curvature
    center_of_curvature_pos = np.array(param_curve(u)) + radius_of_curvature * normal_vector
    center_of_curvature_neg = np.array(param_curve(u)) - radius_of_curvature * normal_vector

    if np.linalg.norm([center_offset_pos - center_of_curvature_pos], 2) < np.linalg.norm([center_offset_neg - center_of_curvature_neg], 2):
        center_of_curvature = center_of_curvature_pos
    else:
        center_of_curvature = center_of_curvature_neg

    return center_of_curvature, radius_of_curvature

clean_data = process_data(data)
x_clean_data = clean_data[0]
y_clean_data = clean_data[1]
u_test = get_param(x_clean_data[260], y_clean_data[260], param_curve)
center_of_curvature, radius_of_curvature = find_center_of_curvature(param_curve, u_test)
print("Center of curvature is ", center_of_curvature)
print("Radius of curvature is ", radius_of_curvature)
# plot_curve(param_curve)
# # plt.scatter(clean_data[0], clean_data[1], color='red', label='Data Points', s=3)
# plt.scatter(center_of_curvature[0], center_of_curvature[1], color='green', label='Center of Curvature')
# plt.scatter(param_curve(u_test)[0], param_curve(u_test)[1], color='red', label='Point on Curve')
# plt.show()

def simulate_river(param_curve, t, precision=1000):
    u = np.linspace(0, 1, precision)
    New_river = np.zeros((len(u), 2))
    for i in range(len(u)):
        u_i = u[i]
        center_of_curvature, radius_of_curvature_OG = find_center_of_curvature(param_curve, u_i)
        speed_coeff = 1/(radius_of_curvature_OG**2)  # Adjust this coefficient to control the speed of change
        # print("Speed coefficient is ", speed_coeff)
        radius_of_curvature = radius_of_curvature_OG * (1 + speed_coeff * t)
        direction_vector = np.array([param_curve(u_i)[0] - center_of_curvature[0], param_curve(u_i)[1] - center_of_curvature[1]])
        direction_vector /= np.linalg.norm(direction_vector)

        New_river[i] = center_of_curvature + radius_of_curvature * direction_vector
    
    New_river_data = pd.DataFrame(New_river,columns=data.columns)
    return New_river_data


New_river_data = simulate_river(param_curve, 1)
New_river_curve = get_curve(New_river_data, None, smooth_factor=0)



def simulate_river_over_time(param_curve, time_step, num_steps):
    time_steps = np.linspace(time_step, time_step * num_steps, num_steps)
    river_paths = []
    for t in time_steps:
        if t == time_steps[0]:
            New_river_data = simulate_river(param_curve, time_step)
        else:
            New_river_data = simulate_river(New_river_curve, time_step)

        New_river_curve = get_curve(New_river_data, None, smooth_factor=10)
        river_paths.append(New_river_curve)

    return river_paths

# plt.plot(New_river_curve(np.linspace(0, 1, 1000))[0], New_river_curve(np.linspace(0, 1, 1000))[1], label='Simulated River Path', color='red')
# plt.scatter(New_river_data[0], New_river_data[1], label='Simulated River Path', color='orange')
# plt.plot(param_curve(np.linspace(0, 1, 1000))[0], param_curve(np.linspace(0, 1, 1000))[1], label='Original River Path')
# plt.axis("equal")
# plt.show()

river_paths = simulate_river_over_time(param_curve, 1, 5)
print(np.shape(river_paths))

plt.plot(river_paths[0](np.linspace(0, 1, 1000))[0], river_paths[0](np.linspace(0, 1, 1000))[1], label='Simulated River Path t=1', color='red')
plt.plot(river_paths[1](np.linspace(0, 1, 1000))[0], river_paths[1](np.linspace(0, 1, 1000))[1], label='Simulated River Path t=2', color='blue')
plt.plot(river_paths[2](np.linspace(0, 1, 1000))[0], river_paths[2](np.linspace(0, 1, 1000))[1], label='Simulated River Path t=3', color='green')
plt.plot(river_paths[3](np.linspace(0, 1, 1000))[0], river_paths[3](np.linspace(0, 1, 1000))[1], label='Simulated River Path t=4', color='purple')
plt.plot(river_paths[4](np.linspace(0, 1, 1000))[0], river_paths[4](np.linspace(0, 1, 1000))[1], label='Simulated River Path t=5', color='orange')
plt.axis("equal")
plt.show()