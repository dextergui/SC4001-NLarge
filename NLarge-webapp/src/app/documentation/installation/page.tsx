import { CodeHighlightTabs } from "@mantine/code-highlight";
import {
  AppShellSection,
  Divider,
  Group,
  rem,
  Stack,
  Text,
  Title,
} from "@mantine/core";

export default function DocumentationInstallation() {
  return (
    <>
      <AppShellSection className="pt-10" bg="bg2">
        <Group justify="center">
          <Stack className="w-3/4 p-2 pb-10">
            <Title
              c="primary"
              ff="monospace"
              size={rem(42)}
              fw="bolder"
              mt="xl"
              ta="left"
            >
              Installation Guide
            </Title>
            <Text c="dimmed" size="lg">
              Getting started with NLarge
            </Text>
          </Stack>
        </Group>
        <Divider my="lg" />
      </AppShellSection>
      <AppShellSection className="pb-10">
        <Group justify="center">
          <Group className="w-2/3">
            <Stack>
              <Text c="primary" size="lg" fw="bolder">
                Get started by installing NLarge
              </Text>
              <Text c="dimmed" size="md">
                You can install NLarge to your preferred python environment.
                Take note that the supported python version is 3.12, use the
                other versions at your own risk.
              </Text>
              <Text c="primary" size="md">
                Install NLarge:
              </Text>
              <CodeHighlightTabs
                code={[
                  {
                    fileName: "shell",
                    code: "pip install NLarge",
                    language: "bash",
                  },
                ]}
                className="w-full rounded-sm outline-1 outline outline-slate-600"
              />
              <Text c="primary" size="lg" fw="bolder">
                Alternative installation
              </Text>
              <Text c="dimmed" size="md">
                If you face issues installing NLarge through pip, you may
                directly clone from our github repository.
              </Text>
              <Text c="primary" size="md">
                Clone NLarge:
              </Text>
              <CodeHighlightTabs
                code={[
                  {
                    fileName: "shell",
                    code: "git clone https://github.com/dextergui/SC4001-NLarge.git",
                    language: "bash",
                  },
                ]}
                className="w-full rounded-sm outline-1 outline outline-slate-600"
              />
              <Divider my={2} />
            </Stack>
          </Group>
        </Group>
      </AppShellSection>
    </>
  );
}
