// PCL lib Functions for processing point clouds

#include "processPointClouds.h"
#include <iostream>

//constructor:
template <typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}

//de-constructor:
template <typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}

template <typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // TODO:: Fill in the function to do voxel grid point reduction and region based filtering
    typename pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    typename pcl::PointCloud<PointT>::Ptr cloud_region(new pcl::PointCloud<PointT>);

    typename pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(filterRes, filterRes, filterRes);
    sor.filter(*cloud_filtered);

    pcl::CropBox<PointT> roiset;
    roiset.setInputCloud(cloud_filtered);
    roiset.setMin(minPoint);
    roiset.setMax(maxPoint);
    roiset.filter(*cloud_region);

    std::vector<int> indices;
    pcl::CropBox<PointT> roof;
    roof.setInputCloud(cloud_region);
    roof.setMin(Eigen::Vector4f(-1.5, -1.7, -1, 1));
    roof.setMax(Eigen::Vector4f(2.6, 1.7, -0.4, 1));
    roof.filter(indices);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    for (int indice : indices)
    {
        inliers->indices.push_back(indice);
    }

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud_region);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_region);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloud_region;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud)
{
    // TODO: Create two new point clouds, one cloud with obstacles and other with segmented plane

    typename pcl::PointCloud<PointT>::Ptr obstCloud(new pcl::PointCloud<PointT>);
    typename pcl::PointCloud<PointT>::Ptr planeCloud(new pcl::PointCloud<PointT>);

    for (int index : inliers->indices)
    {
        planeCloud->points.push_back(cloud->points[index]);
    }

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*obstCloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstCloud, planeCloud);
    return segResult;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    srand(time(NULL));

    int cloudsize = cloud->size();
    float a, b, c, d;
    std::unordered_set<int> maxinliersResult;

    for (int i = 0; i < maxIterations; i++)
    {

        std::unordered_set<int> inliersResult;

        int ind1 = rand() % cloudsize;
        int ind2 = rand() % cloudsize;
        int ind3 = rand() % cloudsize;

        if (ind1 == ind2 || ind2 == ind3 || ind3 == ind1)
        {
            i--;
            continue;
        }
        else
        {
            float x1 = cloud->points[ind1].x;
            float x2 = cloud->points[ind2].x;
            float x3 = cloud->points[ind3].x;

            float y1 = cloud->points[ind1].y;
            float y2 = cloud->points[ind2].y;
            float y3 = cloud->points[ind3].y;

            float z1 = cloud->points[ind1].z;
            float z2 = cloud->points[ind2].z;
            float z3 = cloud->points[ind3].z;

            a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
            b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
            c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
            d = -a * x1 - b * y1 - c * z1;
        }

        for (int j = 0; j < cloudsize; j++)
        {
            float distance = abs(a * cloud->points[j].x + b * cloud->points[j].y + c * cloud->points[j].z + d) / sqrt(a * a + b * b + c * c);
            if (distance < distanceThreshold)
            {
                inliersResult.insert(j);
            }
        }

        if (inliersResult.size() > maxinliersResult.size())
        {
            maxinliersResult = inliersResult;
        }
    }

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    for (int indice : maxinliersResult)
    {
        inliers->indices.push_back(indice);
    }

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers, cloud);

    return segResult;
}

template <typename PointT>
void ProcessPointClouds<PointT>::Proximity(typename pcl::PointCloud<PointT>::Ptr cloud, typename pcl::PointCloud<PointT>::Ptr cluster, std::vector<bool> &processed, KdTree *tree, int index, float distanceTol)
{
    processed[index] = true;
    cluster->points.push_back(cloud->points[index]);
    std::vector<int> nearby = tree->search(cloud->points[index], distanceTol);

    for (int ind : nearby)
    {
        if (!processed[ind])
        {
            Proximity(cloud, cluster, processed, tree, ind, distanceTol);
        }
    }
}

template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, KdTree *tree, float clusterTolerance)
{
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    std::vector<bool> processed(cloud->points.size(), false);

    for (int i = 0; i < cloud->points.size(); ++i)
    {
        if (!processed[i])
        {
            typename pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>);

            Proximity(cloud, cluster, processed, tree, i, clusterTolerance);
            clusters.push_back(cluster);
        }
    }

    return clusters;
}

template <typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}

template <typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII(file, *cloud);
    std::cerr << "Saved " << cloud->points.size() << " data points to " + file << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size() << " data points from " + file << std::endl;

    return cloud;
}

template <typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;
}