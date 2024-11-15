"use client";

import { AppShell, Box, NavLink, rem } from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconBook2, IconCode } from "@tabler/icons-react";
import "@mantine/code-highlight/styles.css";
import { usePathname } from "next/navigation";

export default function DocumentationLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const [opened] = useDisclosure();
  const pathname = usePathname();
  const isActiveLink = (href: string) => {
    const match = pathname.match(href);
    if (match) return true;
    else return false;
  };
  return (
    <AppShell
      padding="md"
      navbar={{ width: 300, breakpoint: "sm", collapsed: { mobile: !opened } }}
    >
      <AppShell.Navbar>
        <Box className="py-4 px-2 w-full">
          <NavLink
            key="documentation"
            label="Documentation"
            leftSection={<IconBook2 />}
            description="A simple guide to use NLarge"
            childrenOffset={rem(14)}
            defaultOpened
            variant="subtle"
            active={isActiveLink("^/documentation/(?!examples/).+")}
          >
            <NavLink
              label="Installation"
              href="/documentation/installation"
              variant="default"
              active={isActiveLink("/documentation/installation")}
            />
            <NavLink
              label="Usage"
              href="/documentation/usage"
              variant="default"
              active={isActiveLink("/documentation/usage")}
            />
            <NavLink
              label="Ready to use Models"
              href="/documentation/models"
              variant="default"
              active={isActiveLink("/documentation/models")}
            />
            <NavLink
              label="Helper functions"
              href="/documentation/helper"
              variant="default"
              active={isActiveLink("/documentation/helper")}
            />
          </NavLink>
          <NavLink
            key="examples"
            label="Usage Examples"
            leftSection={<IconCode />}
            description="NLarge examples and proof of concept"
            childrenOffset={rem(14)}
            defaultOpened
            variant="subtle"
            active={isActiveLink("^/documentation/examples/.+")}
          >
            <NavLink
              label="Random Augmentation"
              href="/documentation/examples/random"
              variant="default"
              active={isActiveLink("/documentation/examples/random")}
            />
            <NavLink
              label="Synonym Augmentation"
              href="/documentation/examples/synonym"
              variant="default"
              active={isActiveLink("/documentation/examples/synonym")}
            />
            <NavLink
              label="LLM Augmentation"
              href="/documentation/examples/llm"
              variant="default"
              active={isActiveLink("/documentation/examples/llm")}
            />
          </NavLink>
        </Box>
      </AppShell.Navbar>
      <AppShell.Main>{children}</AppShell.Main>
    </AppShell>
  );
}
