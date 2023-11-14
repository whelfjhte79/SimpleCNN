#ifndef CHART_H
#define CHART_H

#include<vector>
#include <Windows.h>
#include<string>

std::vector<int> X;
std::vector<int> Y;
std::wstring XLabel;
std::wstring YLabel;

static int minVal(int a, int b) {
    return a >= b ? a : b;
}
class WinApp {
public:

    WinApp(const std::vector<int>& X_, const std::vector<int>& Y_, const std::string& XLabel_, const std::string& YLabel_) {
        hInstance = GetModuleHandle(NULL);
        X = X_;
        Y = Y_;
        XLabel.assign(XLabel_.begin(), XLabel_.end());
        YLabel.assign(YLabel_.begin(), YLabel_.end());
        WNDCLASS wc = {};
        wc.lpfnWndProc = WindowProc;
        wc.hInstance = hInstance;
        wc.lpszClassName = "Visualization";
        RegisterClass(&wc);

        hwnd = CreateWindowExW(
            0,
            L"Visualization",
            L"µö·¯´× ½Ã°¢È­",
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT,
            800, 600,
            NULL,
            NULL,
            hInstance,
            NULL
        );
    }
    ~WinApp() {
        UnregisterClassW(L"WinAPICppExample", hInstance);
    }

    void Run() {
        if (hwnd == NULL) {
            return;
        }

        ShowWindow(hwnd, SW_SHOWDEFAULT);
        UpdateWindow(hwnd);

        MSG msg = {};
        while (GetMessage(&msg, NULL, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
        switch (uMsg) {
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            DrawChart(hdc, X, Y, XLabel, YLabel);

            EndPaint(hwnd, &ps);
            break;
        }
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }
        return 0;
    }

private:
    HWND hwnd;
    HINSTANCE hInstance;

    static void DrawChart(HDC hdc, const std::vector<int>& X, const std::vector<int>& Y, const std::wstring& XLabel, const std::wstring& YLabel) {
        int chartWidth = 600;
        int chartHeight = 400;
        int margin = 50;

        MoveToEx(hdc, margin, margin, NULL);
        LineTo(hdc, margin, margin + chartHeight);
        MoveToEx(hdc, margin, margin + chartHeight, NULL);
        LineTo(hdc, margin + chartWidth, margin + chartHeight);

        TextOutW(hdc, margin + chartWidth / 2 - XLabel.length() * 5, margin + chartHeight + 20, XLabel.c_str(), XLabel.length());
        TextOutW(hdc, margin - 40, margin + chartHeight / 2 - YLabel.length() * 5, YLabel.c_str(), YLabel.length());


        int numPoints = minVal(X.size(), Y.size());
        int xInterval = chartWidth / (numPoints + 1);

        for (int i = 0; i < numPoints; i++) {
            int x = margin + (i + 1) * xInterval;
            int y = margin + chartHeight - Y[i];

            Ellipse(hdc, x - 5, y - 5, x + 5, y + 5);
            if (i < numPoints - 1) {
                int nextX = margin + (i + 2) * xInterval;
                int nextY = margin + chartHeight - Y[i + 1];
                MoveToEx(hdc, x, y, NULL);
                LineTo(hdc, nextX, nextY);
            }
        }
    }
};



#endif// CHART_H