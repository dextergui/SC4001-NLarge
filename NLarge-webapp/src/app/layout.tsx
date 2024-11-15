"use client";
import {
  AppShell,
  AppShellHeader,
  AppShellSection,
  ColorSchemeScript,
  createTheme,
  DEFAULT_THEME,
  MantineProvider,
  mergeMantineTheme,
  virtualColor,
} from "@mantine/core";
import localFont from "next/font/local";
import "./globals.css";
import { NavBar } from "@/components/navbar";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

const theme = mergeMantineTheme(
  DEFAULT_THEME,
  createTheme({
    fontFamily: geistSans.style.fontFamily,
    fontFamilyMonospace: geistMono.style.fontFamily,
    colors: {
      darkPrimary: [
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
      ],
      lightPrimary: [
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
      ],
      darkDimmed: [
        "#BFBEBF",
        "#BFBEBF",
        "#BFBEBF",
        "#BFBEBF",
        "#BFBEBF",
        "#BFBEBF",
        "#BFBEBF",
        "#BFBEBF",
        "#BFBEBF",
        "#BFBEBF",
      ],
      lightDimmed: [
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
      ],
      darkSecondary: [
        "#D0BCFF",
        "#D0BCFF",
        "#D0BCFF",
        "#D0BCFF",
        "#D0BCFF",
        "#D0BCFF",
        "#D0BCFF",
        "#D0BCFF",
        "#D0BCFF",
        "#D0BCFF",
      ],
      lightSecondary: [
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
        "#7363AD",
      ],
      darkTertiary1: [
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
      ],
      lightTertiary1: [
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
        "#AA00FF",
      ],
      darkTertiary2: [
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
      ],
      lightTertiary2: [
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
        "#FF00AA",
      ],
      darkBg2: [
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
        "#28223E",
      ],
      lightBg2: [
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
        "#DED8E1",
      ],
      darkBg: [
        "#1D192B",
        "#1D192B",
        "#1D192B",
        "#1D192B",
        "#1D192B",
        "#1D192B",
        "#1D192B",
        "#1D192B",
        "#1D192B",
        "#1D192B",
      ],
      lightBg: [
        "#FFFFFF",
        "#FFFFFF",
        "#FFFFFF",
        "#FFFFFF",
        "#FFFFFF",
        "#FFFFFF",
        "#FFFFFF",
        "#FFFFFF",
        "#FFFFFF",
        "#FFFFFF",
      ],
      primary: virtualColor({
        name: "primary",
        dark: "darkPrimary",
        light: "lightPrimary",
      }),
      dimmed: virtualColor({
        name: "dimmed",
        light: "lightDimmed",
        dark: "darkDimmed",
      }),
      secondary: virtualColor({
        name: "secondary",
        dark: "darkSecondary",
        light: "lightSecondary",
      }),
      tertiary1: virtualColor({
        name: "tertiary1",
        dark: "darkTertiary1",
        light: "lightTertiary1",
      }),
      tertiary2: virtualColor({
        name: "tertiary2",
        dark: "darkTertiary2",
        light: "lightTertiary2",
      }),
      bg: virtualColor({
        name: "bg",
        dark: "darkBg",
        light: "lightBg",
      }),
      bg2: virtualColor({
        name: "bg2",
        dark: "darkBg2",
        light: "lightBg2",
      }),
    },
    primaryColor: "secondary",
  }),
);

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <ColorSchemeScript />
      </head>
      <body className="antialiased">
        <MantineProvider
          defaultColorScheme="dark"
          theme={theme}
          withGlobalClasses
        >
          <AppShell header={{ height: 60 }} padding="md">
            <AppShellHeader>
              <NavBar />
            </AppShellHeader>
            <AppShellSection>{children}</AppShellSection>
          </AppShell>
        </MantineProvider>
      </body>
    </html>
  );
}
