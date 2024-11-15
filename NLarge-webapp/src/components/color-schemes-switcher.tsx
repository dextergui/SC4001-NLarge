"use client";

import {
  useMantineColorScheme,
  Group,
  Switch,
  useMantineTheme,
  rem,
} from "@mantine/core";
import { MoonIcon, SunIcon } from "@radix-ui/react-icons";

export function ColorSchemesSwitcher() {
  const { colorScheme, toggleColorScheme } = useMantineColorScheme({
    keepTransitions: true,
  });
  const theme = useMantineTheme();
  const isDark = colorScheme === "dark";

  const sunIcon = (
    <SunIcon
      style={{ width: rem(16), height: rem(16) }}
      stroke="2.5"
      color={theme.colors.blue[6]}
    />
  );

  const moonIcon = (
    <MoonIcon
      style={{ width: rem(16), height: rem(16) }}
      stroke="2.5"
      color={theme.colors.yellow[4]}
    />
  );

  return (
    <Group>
      {/* <Button onClick={() => setColorScheme("light")}>Light</Button>
      <Button onClick={() => setColorScheme("dark")}>Dark</Button>
      <Button onClick={() => setColorScheme("auto")}>Auto</Button>
      <Button onClick={clearColorScheme}>Clear</Button> */}
      <Switch
        size="md"
        color="dark.4"
        onLabel={moonIcon}
        offLabel={sunIcon}
        checked={isDark}
        onChange={() => toggleColorScheme()}
      />
    </Group>
  );
}
