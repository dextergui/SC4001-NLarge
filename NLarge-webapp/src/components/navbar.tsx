"use client";

import {
  Anchor,
  Button,
  Group,
  Text,
  useMantineColorScheme,
} from "@mantine/core";
import { Logo } from "./logo";
import { ColorSchemesSwitcher } from "./color-schemes-switcher";
import { GitHubLogoIcon } from "@radix-ui/react-icons";
import { IconBook2 } from "@tabler/icons-react";
import Link from "next/link";

export function NavBar() {
  const { colorScheme } = useMantineColorScheme({
    keepTransitions: true,
  });
  const isDark = colorScheme === "dark";
  return (
    <Group className="h-full px-16 justify-between">
      <Anchor href="/" underline="never">
        <Group className="w-auto dark:fill-blue transform transition-transform duration-300 hover:scale-105">
          <Logo
            width="50"
            height="50"
            stroke="4"
            color={isDark ? "#D0BCFF" : "#7363AD"}
          />
          <Text size="xl" fw={700} c="secondary">
            NLarge
          </Text>
        </Group>
      </Anchor>
      <Group>
        <ColorSchemesSwitcher />
        <Link href="https://github.com/dextergui/SC4001-NLarge">
          <Button variant="outline" size="compact-md" color="secondary">
            <GitHubLogoIcon />
          </Button>{" "}
        </Link>
        <Link href="/documentation">
          <Button variant="outline" size="compact-md" color="secondary">
            <IconBook2 />
          </Button>
        </Link>
      </Group>
    </Group>
  );
}
