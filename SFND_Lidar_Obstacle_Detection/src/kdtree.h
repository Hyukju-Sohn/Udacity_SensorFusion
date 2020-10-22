/* \author Aaron Brown */
// Quiz on implementing kd tree

#pragma once
#include "render/render.h"

// Structure to represent node of kd tree

struct Node
{
	pcl::PointXYZI point;
	int id;
	Node *left;
	Node *right;

	Node(pcl::PointXYZI arr, int setId)
		: point(arr), id(setId), left(NULL), right(NULL)
	{
	}
};

struct KdTree
{
	Node *root;

	KdTree()
		: root(NULL)
	{
	}

	void insertHelp(Node *&node, int dim, pcl::PointXYZI point, int id)
	{

		dim = dim % 2;

		if (node == NULL)
		{
			node = new Node(point, id);
		}
		else if (dim == 0)
		{
			if (point.x > node->point.x)
			{
				insertHelp(node->right, dim + 1, point, id);
			}
			else
			{
				insertHelp(node->left, dim + 1, point, id);
			}
		}
		else
		{
			if (point.y > node->point.y)
			{
				insertHelp(node->right, dim + 1, point, id);
			}
			else
			{
				insertHelp(node->left, dim + 1, point, id);
			}
		}
	}

	void insert(pcl::PointXYZI point, int id)
	{
		insertHelp(root, 0, point, id);
	}

	void searchHelp(Node *&node, pcl::PointXYZI target, float distanceTol, std::vector<int> &ids, int dim)
	{

		dim = dim % 2;

		if (node != NULL)
		{
			if (abs(target.x - node->point.x) <= distanceTol && abs(target.y - node->point.y) <= distanceTol)
			{
				float distance = sqrt((target.x - node->point.x) * (target.x - node->point.x) + (target.y - node->point.y) * (target.y - node->point.y));

				if (distance <= distanceTol)
				{
					ids.push_back(node->id);
				}
			}

			if (dim == 0)
			{
				if (target.x - distanceTol < node->point.x)
				{
					searchHelp(node->left, target, distanceTol, ids, dim + 1);
				}
				if (target.x + distanceTol > node->point.x)
				{
					searchHelp(node->right, target, distanceTol, ids, dim + 1);
				}
			}
			else
			{
				if (target.y - distanceTol < node->point.y)
				{
					searchHelp(node->left, target, distanceTol, ids, dim + 1);
				}
				if (target.y + distanceTol > node->point.y)
				{
					searchHelp(node->right, target, distanceTol, ids, dim + 1);
				}
			}
		}
	}

	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(pcl::PointXYZI target, float distanceTol)
	{
		std::vector<int> ids;
		searchHelp(root, target, distanceTol, ids, 0);

		return ids;
	}
};
