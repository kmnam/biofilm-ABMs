/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/12/2025
 */

#include <Eigen/Dense>
#include "../include/indices.hpp"
#include "../include/utils.hpp"

template <typename T>
Array<T, Dynamic, 10> trackLineage(const std::vector<Array<T, Dynamic, Dynamic> >& cells, 
                                   std::vector<int>& parents, std::vector<T>& times, 
                                   const int cell_id)
{
    // Get the ancestors of the specified cell 
    std::vector<int> ancestors = getAncestry(cell_id, parents);

    // Start with the very first ancestor (skipping over the initial -1)
    int curr_idx = 1; 
    int curr_ancestor = ancestors[1];
   
    // Run through the simulation frames ... 
    Array<T, Dynamic, 10> trajectory(cells.size(), 10); 
    int i = 0;  
    for (auto it = cells.begin(); it != cells.end(); ++it)
    {
        // Look for the current ancestor
        Array<T, Dynamic, Dynamic> cells_i = *it;  
        const int ncells = cells_i.rows();
        bool found_ancestor = false;
        int j = 0;  
        while (!found_ancestor && j < ncells)
        {
            if (cells_i(j, __colidx_id) == curr_ancestor)
                found_ancestor = true;
            else
                j++; 
        }

        // If the current ancestor is not in the population, look for the 
        // next ancestor 
        if (!found_ancestor)
        {
            curr_idx++;
            curr_ancestor = ancestors[curr_idx]; 
            j = 0;  
            while (!found_ancestor && j < ncells)
            {
                if (cells_i(j, __colidx_id) == curr_ancestor)
                    found_ancestor = true;
                else
                    j++; 
            }
        }

        // If the next ancestor could not be found, then throw an exception  
        if (!found_ancestor)
            throw std::runtime_error("Could not find cell in specified lineage");

        // Extract the cell's position and orientation
        trajectory(i, 0) = times[i]; 
        trajectory(i, 1) = cells_i(j, __colidx_id); 
        trajectory(i, Eigen::seq(2, 4)) = cells_i(j, __colseq_r); 
        trajectory(i, Eigen::seq(5, 7)) = cells_i(j, __colseq_n);
        trajectory(i, 8) = cells_i(j, __colidx_half_l);  
        trajectory(i, 9) = cells_i(j, __colidx_group); 
        i++; 
    }

    return trajectory; 
}

int main(int argc, char** argv)
{
    // Parse input directory of simulation files 
    std::string indir = argv[1]; 
    auto filenames = parseDir(indir); 
    std::vector<std::string> frame_filenames = filenames.first; 
    std::string lineage_filename = filenames.second;

    // Parse each simulation file ... 
    std::vector<Array<double, Dynamic, Dynamic> > simulation; 
    std::vector<double> times; 
    for (auto& filename : frame_filenames)
    {
        auto result = readCells<double>(filename);
        simulation.push_back(result.first);
        times.push_back(std::stod(result.second["t_curr"]));  
    }

    // Parse the lineage file 
    std::vector<int> parents; 
    auto lineage = readLineage(lineage_filename);
    parents.resize(lineage.size()); 
    for (const auto& pair : lineage)
        parents[pair.first] = pair.second;

    // Choose cells to track
    std::string option = argv[3];
    std::vector<int> cell_ids;
    if (option == "-l")
    {
        // Track lineages of the specified cells 
        std::string ids = argv[4]; 
        std::stringstream ss; 
        ss << ids;
        std::string token;  
        while (std::getline(ss, token, ','))
            cell_ids.push_back(std::stoi(token)); 
    }
    else if (option == "-r")
    {
        // Track lineages of the given collection of randomly sampled cells 
        const int seed = std::stoi(argv[4]);
        const int n_sample = std::stoi(argv[5]);
        Array<double, Dynamic, Dynamic> cells_final = simulation[simulation.size() - 1];  
        const int n_final = cells_final.rows(); 
        boost::random::mt19937 rng(seed); 
        std::vector<int> idx = sampleWithoutReplacement(n_final, n_sample, rng);
        for (const int& i : idx)
            cell_ids.push_back(static_cast<int>(cells_final(i, __colidx_id))); 
    } 
    else 
    {
        std::stringstream ss; 
        ss << "Invalid option specified: " << option; 
        throw std::runtime_error(ss.str()); 
    }

    // Extract the trajectory from each cell
    std::vector<Array<double, Dynamic, 10> > trajectories; 
    for (int i = 0; i < cell_ids.size(); ++i)
        trajectories.push_back(
            trackLineage<double>(simulation, parents, times, cell_ids[i])
        );

    // Write each trajectory to a separate file
    std::string outprefix = argv[2]; 
    for (int i = 0; i < cell_ids.size(); ++i)
    {
        std::stringstream ss; 
        ss << outprefix << "_traj" << cell_ids[i] << ".txt";
        std::ofstream outfile(ss.str());
        outfile << std::setprecision(10); 
        for (int j = 0; j < trajectories[i].rows(); ++j)
        {
            for (int k = 0; k < 9; ++k)
            {
                outfile << trajectories[i](j, k) << '\t'; 
            }
            outfile << trajectories[i](j, 9) << std::endl; 
        }  
        outfile.close();  
    }
    
    return 0; 
}
