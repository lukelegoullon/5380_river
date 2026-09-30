clear; close all; clc;

RAW_data_path1 = readmatrix("Path1.txt");

Lat_path1 = RAW_data_path1(:,2);
Lon_path1 = RAW_data_path1(:,3);

[x_path1,y_path1,utmzone_path1] = deg2utm(Lat_path1,Lon_path1);

UTM_path1 = [x_path1 y_path1];

% figure;
% plot(x_path1,y_path1)

RAW_data_path2 = readmatrix("Path2.txt");

Lat_path2 = RAW_data_path2(:,2);
Lon_path2 = RAW_data_path2(:,3);

[x_path2,y_path2,utmzone_path2] = deg2utm(Lat_path2,Lon_path2);

UTM_path2 = [x_path2 y_path2];

% figure; 
% plot(x_path2,y_path2)


writematrix(UTM_path1, 'Path1_UTM.csv')
writematrix(UTM_path2, 'Path2_UTM.csv')