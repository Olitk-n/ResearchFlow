import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ResearchFlow · AI 科研自动化",
  description: "从前沿论文到可复现实验与投稿草稿的本地科研工作台",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}

