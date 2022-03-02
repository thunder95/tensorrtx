//
// Created by hl on 22-3-1.
//

#ifndef TRTPOSE_COVER_TABLE_H
#define TRTPOSE_COVER_TABLE_H

#pragma once

#include <memory>
#include <vector>


class TrtPoseCoverTable {
public:
    TrtPoseCoverTable(int nrows, int ncols) : nrows(nrows), ncols(ncols) {
        char delay[256] = {0}; // FIXME: 这里只是象征性的延时一下，因为出现过多次rows.resize(nrows)崩
        rows.resize(nrows);
        cols.resize(ncols);
    }

    inline void coverRow(int row) {
        rows[row] = 1;
    }

    inline void coverCol(int col) {
        cols[col] = 1;
    }

    inline void uncoverRow(int row) {
        rows[row] = 0;
    }

    inline void uncoverCol(int col) {
        cols[col] = 0;
    }

    inline bool isCovered(int row, int col) const {
        return rows[row] || cols[col];
    }

    inline bool isRowCovered(int row) const {
        return rows[row];
    }

    inline bool isColCovered(int col) const {
        return cols[col];
    }

    inline void clear() {
        for (int i = 0; i < nrows; i++) {
            uncoverRow(i);
        }
        for (int j = 0; j < ncols; j++) {
            uncoverCol(j);
        }
    }

    const int nrows;
    const int ncols;

private:
    std::vector<bool> rows;
    std::vector<bool> cols;
};


#endif //TRTPOSE_COVER_TABLE_H
