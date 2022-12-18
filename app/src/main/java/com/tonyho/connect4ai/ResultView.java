// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package com.tonyho.connect4ai;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;


public class ResultView extends View {

    private final static int TEXT_X = 40;
    private final static int TEXT_Y = 35;
    private final static int TEXT_WIDTH = 260;
    private final static int TEXT_HEIGHT = 50;

    private Paint mPaintRectangle;
    private Paint mPaintText;
    private ArrayList<Result> mResults;
    private int bestMove = -1;  //result index for best column
    private int[][] board = new int[6][7];
    private boolean isRedFirst = true;

    public ResultView(Context context) {
        super(context);
    }

    public ResultView(Context context, AttributeSet attrs){
        super(context, attrs);
        mPaintRectangle = new Paint();
        mPaintRectangle.setColor(Color.YELLOW);
        mPaintText = new Paint();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (mResults == null) return;
        for (int i = 0; i < mResults.size(); ++i) {
            Result result = mResults.get(i);

            mPaintRectangle.setStrokeWidth(2);
            mPaintRectangle.setStyle(Paint.Style.STROKE);
            //high light column that should move next
            if(i == bestMove)
            {
                mPaintRectangle.setColor(Color.MAGENTA);
            }
            else
            {
                mPaintRectangle.setColor(Color.YELLOW);
            }
            canvas.drawRect(result.rect, mPaintRectangle);

            /*
            Path mPath = new Path();
            RectF mRectF = new RectF(result.rect.left, result.rect.top, result.rect.left + TEXT_WIDTH,  result.rect.top + TEXT_HEIGHT);
            mPath.addRect(mRectF, Path.Direction.CW);
            mPaintText.setColor(Color.MAGENTA);
            canvas.drawPath(mPath, mPaintText);

             */

            mPaintText.setColor(Color.WHITE);
            mPaintText.setStrokeWidth(0);
            mPaintText.setStyle(Paint.Style.FILL);
            mPaintText.setTextSize(16);
            canvas.drawText(String.format("%s %.2f", CameraActivity.mClasses[result.classIndex], result.score), result.rect.left + TEXT_X, result.rect.top + TEXT_Y, mPaintText);
        }
    }

    public void setResults(ArrayList<Result> results) {

        mResults = results;

        ArrayList<Result> columns = new ArrayList<>();
        ArrayList<Result> redPieces = new ArrayList<>();
        ArrayList<Result> yellowPieces = new ArrayList<>();

        for (int i = 0; i < mResults.size(); ++i) {

            //class index 1 = column
            if(mResults.get(i).classIndex == 1)
            {
                columns.add(mResults.get(i));
            }
            else if(mResults.get(i).classIndex == 2)
            {
                redPieces.add(mResults.get(i));
            }
            else if(mResults.get(i).classIndex == 3)
            {
                yellowPieces.add(mResults.get(i));
            }
        }

        //no prediction if not enough columns, early out
        if(columns.size() < 7)
        {
            bestMove = -1;
            return;
        }
        else if(columns.size() > 7) //get only the top 7 columns
        {
            columns.sort((r1,r2) -> Float.compare(r1.score, r2.score));

            while(columns.size() > 7)
            {
                columns.remove(0);
            }
        }

        //sort columns from left to right
        columns.sort((r1,r2) -> Float.compare(
                (r1.rect.right - r1.rect.left)/2 + r1.rect.left,
                (r2.rect.right - r2.rect.left)/2 + r2.rect.left));

        //reset board
        for(int i = 0; i < 6; ++i)
        {
            for(int j = 0; j < 7; ++j)
            {
                board[i][j] = 0;
            }
        }

        board = setBoard(redPieces, columns, board, 1);
        board = setBoard(yellowPieces, columns, board, -1);

        //if there's a hole in the board/floating pieces, invalidate pieces above the hole
        for(int i = 0; i < 7; ++i)
        {
            boolean foundHole = false;
            for(int j = 0; j < 6; ++j)
            {
                if(foundHole)
                {
                    board[j][i] = 0;
                }
                else if(board[j][i] == 0)
                {
                    foundHole = true;
                }
            }
        }

        int redCount = 0;
        int yellowCount = 0;
        for(int i = 0; i < 6; ++i)
        {
            for(int j = 0; j < 7; ++j)
            {
                if(board[i][j] == 1)
                {
                    redCount += 1;
                }
                else if(board[i][j] == -1)
                {
                    yellowCount += 1;
                }
            }
        }

        //early out if no pieces to predict
        if( redCount == 0 && yellowCount == 0)
        {
            bestMove = -1;
            return;
        }

        if(redCount == 1 && yellowCount == 0)
            isRedFirst = true;
        else if(redCount == 0 && yellowCount == 1)
            isRedFirst = false;

        if(yellowCount > redCount)
            bestMove = getBestMove(board, true);
        else if(redCount > yellowCount)
            bestMove = getBestMove(board, false);
        else    //tie, then first player moves
            bestMove = getBestMove(board, isRedFirst);

        //convert from best column to best result index (for highlighting rect)
        Rect bestRect = columns.get(bestMove).rect;

        for(int i = 0; i < mResults.size(); ++i)
        {
            Result result = mResults.get(i);

            if(result.classIndex == 1 && result.rect == bestRect)
            {
                bestMove = i;
                break;
            }
        }
    }

    int[][] setBoard(ArrayList<Result> colorPieces, ArrayList<Result> columns, int[][] board, int pieceLabel)
    {
        for(int i = 0; i < colorPieces.size();++i)
        {
            Rect rRect = colorPieces.get(i).rect;

            int bestCol = 0;
            int bestDist = Integer.MAX_VALUE;

            int x = (rRect.right - rRect.left)/2 + rRect.left;
            int y = (rRect.bottom - rRect.top)/2 + rRect.top;

            //find best matching column
            for(int j = 0; j < columns.size();++j)
            {
                Rect colRect = columns.get(j).rect;
                int colMid = (colRect.right - colRect.left)/2 + colRect.left;

                int diff = Math.abs(colMid - x);
                if(diff < bestDist)
                {
                    bestDist = diff;
                    bestCol = j;
                }
            }

            Rect colRect = columns.get(bestCol).rect;
            int height = (colRect.bottom - colRect.top) / 6;
            int heights[] = new int[]{
                    colRect.bottom - height/2,
                    colRect.bottom - (height/2*3),
                    colRect.bottom - (height/2*5),
                    colRect.bottom - (height/2*7),
                    colRect.bottom - (height/2*9),
                    colRect.bottom - (height/2*11),
            };

            int bestHeight = 0;
            int bestHeightDist = Integer.MAX_VALUE;
            for(int j = 0; j < heights.length;++j) {
                int diff = Math.abs(heights[j] - y);
                if (diff < bestHeightDist) {
                    bestHeightDist = diff;
                    bestHeight = j;
                }
            }

            board[bestHeight][bestCol] = pieceLabel;
        }

        return board;
    }

    int getBestMove(int[][] board, boolean isRed)
    {
        minimaxAgent mma = new minimaxAgent(3);
        State s=new State(6,7);

        int flipper = isRed ? 1 : -1;
        for(int i = 0; i < 6; ++i)
        {
            for(int j = 0; j < 7; ++j)
            {
                char c = '.';
                if(board[i][j] == (1*flipper))
                    c = 'O';
                else if (board[i][j] == (-1*flipper))
                    c = 'X';

                //expects top to be 0, but we use bot as 0
                //convert 5->0, 4->1, etc. 0->5
                s.board[5-i][j] = c;
            }
        }

        int bestAction = 3;
        try {
            bestAction = mma.getAction(s);
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }

        return bestAction;
    }
}
