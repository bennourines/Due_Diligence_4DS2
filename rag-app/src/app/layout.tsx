import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from './providers';
import { ThemeProvider } from "next-themes";
import Aoscompo from "@/utils/aos";

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: 'LST',
  description: 'A ChatGPT-like interface for document Q&A',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}