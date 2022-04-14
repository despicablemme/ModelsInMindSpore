"""
get matrix for calculating

Note::::::    vc_mat cv_mat 可优化
"""
import numpy as np


def number_edges(H, n):
    """
    number edges from 0 to (num_edges - 1)
    return: number matrix of edges, and numbers of edges
    """
    num_mat = np.zeros_like(H, dtype='float64')
    k = 0
    for i in range(n):
        ones = np.where(H[:, i] == 1)[0]  # 按列找1
        for j in range(len(ones)):  # 每个1按列顺序编号
            num_mat[ones[j], i] = k
            k += 1
    return num_mat, k


def get_vc_mat(H, m, n, num_edges_mat, num_edges):
    """
    Get message matrix for calculating messages on all edges from variable nodes to check nodes.
    Mat with shape (num_edges, num_edges).
    In vc_mat, one column represents the one edge, in one column, values are all 0 expect for
        the other edges which connected with the variable node(value 1 in a column from H).
    Search edges in columns of H.
    """
    vc_mat = np.zeros(shape=(num_edges, num_edges))
    for col in range(n):
        for row in range(m):

            if H[row, col] == 1:  # find the edge, then locate other edges in the column.
                edge_count = int(num_edges_mat[row, col])
                edge_locate = np.argwhere(H[:, col] == 1)  # edge_locate represents the row of the edge.

                for locate in edge_locate:
                    if locate[0] == row:  # when cal the edge message, ignore message from itself.
                        pass
                    else:
                        number_edge = num_edges_mat[locate[0], col]
                        vc_mat[int(number_edge), edge_count] = 1
    return vc_mat


def get_cv_mat(H, m, n, num_edges_mat, num_edges):
    """
    Get message matrix for calculating messages on all edges from check nodes to variable nodes.
    Mat with shape (num_edges, num_edges).
    In cv_mat, one column represents the one edge, in one column, values are all 0 expect for
        the other edges which connected with the check node(value 1 in a row from H).
    Search edges in columns of H.
    """
    cv_mat = np.zeros(shape=(num_edges, num_edges))
    for col in range(n):
        for row in range(m):

            if H[row, col] == 1:  # find the edge, then locate other edges in the column.
                edge_count = int(num_edges_mat[row, col])
                edge_locate = np.argwhere(H[row, :] == 1)  # edge_locate represents the col of the edge.

                for locate in edge_locate:
                    if locate[0] == col:  # when cal the edge message, ignore message from itself.
                        pass
                    else:
                        number_edge = num_edges_mat[row, locate[0]]
                        cv_mat[int(number_edge), edge_count] = 1
    return cv_mat


def get_llr_mat(n, num_edges_mat, num_edges):
    """
    Get LLR matrix for calculating the initial LLR value on each edge.
    """
    llr_mat = np.zeros((n, num_edges))
    for i in range(num_edges):
        col = np.argwhere(num_edges_mat == i)[0][1]  # col value represents the v-node that this edge connected
        llr_mat[col, i] = 1
    return llr_mat


def get_mats(H, m, n):
    """

    """
    num_edges_mat, num_edges = number_edges(H, n)
    vc_mat = get_vc_mat(H, m, n, num_edges_mat, num_edges)
    cv_mat = get_cv_mat(H, m, n, num_edges_mat, num_edges)
    llr_mat = get_llr_mat(n, num_edges_mat, num_edges)
    return vc_mat, cv_mat, llr_mat, np.transpose(llr_mat), num_edges
