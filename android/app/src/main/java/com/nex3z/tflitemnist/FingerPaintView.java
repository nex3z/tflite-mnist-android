package com.nex3z.tflitemnist;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class FingerPaintView extends View {
    private static final String LOG_TAG = FingerPaintView.class.getSimpleName();
    private static final float TOUCH_TOLERANCE = 4;
    private static final float PEN_SIZE = 48;

    private Paint mPenPaint = new Paint();
    private final Path mPath = new Path();
    private Bitmap mDrawingBitmap;
    private Canvas mDrawingCanvas;
    private final Paint mDrawingPaint = new Paint(Paint.DITHER_FLAG);
    private float mX, mY;
    private boolean empty = true;

    public FingerPaintView(Context context) {
        this(context, null);
    }

    public FingerPaintView(Context context, AttributeSet attrs) {
        super(context, attrs);
        mPenPaint.setAntiAlias(true);
        mPenPaint.setDither(true);
        mPenPaint.setColor(Color.BLACK);
        mPenPaint.setStyle(Paint.Style.STROKE);
        mPenPaint.setStrokeJoin(Paint.Join.ROUND);
        mPenPaint.setStrokeCap(Paint.Cap.ROUND);
        mPenPaint.setStrokeWidth(PEN_SIZE);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        mDrawingBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        mDrawingCanvas = new Canvas(mDrawingBitmap);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawBitmap(mDrawingBitmap, 0, 0, mDrawingPaint);
        canvas.drawPath(mPath, mPenPaint);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        empty = false;
        float x = event.getX();
        float y = event.getY();
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                onTouchStart(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_MOVE:
                onTouchMove(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_UP:
                onTouchUp();
                performClick();
                invalidate();
                break;
        }
        return true;
    }

    @Override
    public boolean performClick() {
        return super.performClick();
    }

    public void clear() {
        mPath.reset();
        mDrawingBitmap = Bitmap.createBitmap(mDrawingBitmap.getWidth(), mDrawingBitmap.getHeight(),
                Bitmap.Config.ARGB_8888);
        mDrawingCanvas = new Canvas(mDrawingBitmap);
        empty = true;
        invalidate();
    }

    public Bitmap exportToBitmap() {
        Bitmap bitmap = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        Drawable bgDrawable = getBackground();
        if (bgDrawable!=null) {
            bgDrawable.draw(canvas);
        } else {
            canvas.drawColor(Color.WHITE);
        }
        draw(canvas);
        return bitmap;
    }

    public Bitmap exportToBitmap(int width, int height) {
        Bitmap rawBitmap = exportToBitmap();
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(rawBitmap, width, height, false);
        rawBitmap.recycle();
        return scaledBitmap;
    }

    public boolean isEmpty() {
        return empty;
    }

    private void onTouchStart(float x, float y) {
        mPath.reset();
        mPath.moveTo(x, y);
        mX = x;
        mY = y;
    }
    private void onTouchMove(float x, float y) {
        float dx = Math.abs(x - mX);
        float dy = Math.abs(y - mY);
        if (dx >= TOUCH_TOLERANCE || dy >= TOUCH_TOLERANCE) {
            mPath.quadTo(mX, mY, (x + mX) / 2, (y + mY) / 2);
            mX = x;
            mY = y;
        }
    }
    private void onTouchUp() {
        mPath.lineTo(mX, mY);
        mDrawingCanvas.drawPath(mPath, mPenPaint);
        mPath.reset();
    }
}
