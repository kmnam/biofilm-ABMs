/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/8/2025
 */

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <filesystem>
#include <unordered_map>
#include <boost/container_hash/hash.hpp>
#include <Eigen/Dense>

int main(int argc, char** argv)
{
    // Parse input arguments 
    std::string indir = argv[1];
    std::string outfilename = argv[2];

    // Get a list of all the files in the input directory 
    std::filesystem::path indir_(indir);
    if (!std::filesystem::is_directory(indir_))
    {
        throw std::runtime_error("Specified directory does not exist"); 
    }
    std::vector<std::string> filenames; 
    for (const auto& entry : std::filesystem::directory_iterator(indir_))
    {
        if (std::filesystem::is_regular_file(entry.status()))
        {
            filenames.push_back(entry.path().string());  
        } 
    }

    // First open each file and parse the second line to get the corresponding
    // value of phi
    std::vector<std::string> phi;
    std::ifstream infile;
    std::string line, token;  
    for (const auto& filename : filenames)
    {
        // Skip the first line and parse the second line in the file ... 
        infile.open(filename); 
        std::getline(infile, line);
        std::getline(infile, line); 

        // ... and extract the third entry in the line 
        std::stringstream ss; 
        ss << line; 
        std::getline(ss, token, '\t');
        std::getline(ss, token, '\t'); 
        std::getline(ss, token, '\t'); 
        phi.push_back(token);
        infile.close();  
    }

    // Sort the phi values and get their indices 
    std::unordered_map<std::string, int> phi_idx;
    std::sort(
        phi.begin(), phi.end(),
        [](const std::string& a, const std::string& b) -> bool
        {
            return std::stod(a) < std::stod(b); 
        }
    );
    for (int i = 0; i < phi.size(); ++i)
        phi_idx[phi[i]] = i;

    // Then reopen just the first file to collect the overlap distances and 
    // radii of curvature
    std::vector<std::string> Rx, delta;  
    std::unordered_map<std::string, int> Rx_idx, delta_idx;  
    infile.open(filenames[0]);
    std::getline(infile, line);    // Skip the first line
    int Rx_i = 0;  
    int delta_i = 0; 
    while (std::getline(infile, line))
    {
        // Extract the first entry in the line (smaller principal radius of curvature)
        std::stringstream ss; 
        ss << line; 
        std::getline(ss, token, '\t');
        std::string Rx_ = token; 

        // Then extract the fourth entry in the line (overlap distance)
        std::getline(ss, token, '\t'); 
        std::getline(ss, token, '\t'); 
        std::getline(ss, token, '\t');
        std::string delta_ = token; 

        // Gather these entries if they haven't been previously encountered 
        if (Rx_idx.find(Rx_) == Rx_idx.end())
        { 
            Rx.push_back(Rx_);
            Rx_idx[Rx_] = Rx_i;
            Rx_i++;
        } 
        if (delta_idx.find(delta_) == delta_idx.end())
        {
            delta.push_back(delta_); 
            delta_idx[delta_] = delta_i; 
            delta_i++; 
        }
    }
    infile.close();

    // Now reopen each file and collect the input values ...
    std::string Rx1_curr, Rx2_curr, phi_curr, delta_curr, force_curr, radius_curr, gamma;
    int Rx1_i = 0;
    int Rx2_i = 0; 
    int phi_i = 0; 
    delta_i = 0;
    int file_i = 0;  
    std::unordered_map<std::array<int, 4>, 
                       std::array<std::string, 2>,
                       boost::hash<std::array<int, 4> > > forces; 
    for (const auto& filename : filenames)
    {
        infile.open(filename); 

        // Parse the first line, which gives the surface adhesion energy density 
        std::getline(infile, line); 
        gamma = line;

        // Parse each subsequent line ... 
        while (std::getline(infile, line))
        {
            // Parse each entry in the line 
            std::stringstream ss; 
            ss << line; 
            std::getline(ss, token, '\t');    // First principal radius of curvature
            Rx1_curr = token; 
            Rx1_i = Rx_idx[token];  
            std::getline(ss, token, '\t');    // Second principal radius of curvature
            Rx2_curr = token; 
            Rx2_i = Rx_idx[token];
            std::getline(ss, token, '\t');    // Phi (cell-cell angle) 
            phi_curr = token;
            phi_i = phi_idx[token];  
            std::getline(ss, token, '\t');    // Overlap distance
            delta_curr = token; 
            delta_i = delta_idx[token]; 
            std::getline(ss, token, '\t');    // JKR force magnitude
            force_curr = token; 
            std::getline(ss, token, '\t');    // Contact radius 
            radius_curr = token;
            std::array<int, 4> key = {Rx1_i, Rx2_i, phi_i, delta_i}; 
            std::array<std::string, 2> values = {force_curr, radius_curr}; 
            forces[key] = values;
        }

        infile.close();
        file_i++;  
    }

    // Write the merged table to file 
    std::ofstream outfile(outfilename);
    outfile << gamma << std::endl;  
    for (int i = 0; i < Rx.size(); ++i)
    {
        for (int j = i; j < Rx.size(); ++j)
        {
            for (int k = 0; k < phi.size(); ++k)
            {
                for (int m = 0; m < delta.size(); ++m)
                {
                    std::array<int, 4> key = {i, j, k, m}; 
                    outfile << Rx[i] << '\t'
                            << Rx[j] << '\t'
                            << phi[k] << '\t'
                            << delta[m] << '\t'
                            << forces[key][0] << '\t'
                            << forces[key][1] << std::endl; 
                }
            }
        }
    } 

    return 0; 
}
