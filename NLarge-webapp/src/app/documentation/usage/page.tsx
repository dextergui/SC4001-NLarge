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
import Link from "next/link";

export default function DocumentationUsage() {
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
              Usage Guide
            </Title>
            <Text c="dimmed" size="lg">
              Beginner usage guide to using NLarge
            </Text>
          </Stack>
        </Group>
        <Divider my="lg" />
      </AppShellSection>
      <AppShellSection className=" pb-10">
        <Group justify="center">
          <Group className="w-2/3">
            <Stack>
              <Text c="primary" size="lg" fw="bolder">
                Importing the library
              </Text>
              <Text c="dimmed" size="md">
                Before using NLarge, start by importing the library. For this
                example, we will be importing the synonym augmenter, you may
                find the usage guide of other augmentation types:
              </Text>
              <Link
                href="/documentation/examples/random"
                className="flex ml-6 hover:text-primary"
              >
                -
                <Text className="ml-2 underline" ff="monospace" size="sm">
                  Random Augmenter
                </Text>
              </Link>
              <Link
                href="/documentation/examples/synonym"
                className="flex ml-6 hover:text-primary"
              >
                -
                <Text className="ml-2 underline" ff="monospace" size="sm">
                  Synonym Augmenter
                </Text>
              </Link>
              <Link
                href="/documentation/examples/llm"
                className="flex ml-6 hover:text-primary"
              >
                -
                <Text className="ml-2 underline" ff="monospace" size="sm">
                  Large Language Model (LLM) Augmenter
                </Text>
              </Link>
              <Text c="primary" size="md">
                Import NLarge Synonym Augmenter:
              </Text>
              <CodeHighlightTabs
                code={[
                  {
                    fileName: "python",
                    code: "from NLarge.synonym import SynonymAugmenter",
                    language: "python",
                  },
                ]}
                className="w-full rounded-sm outline-1 outline outline-slate-600"
              />
              <Text c="primary" size="lg" fw="bolder">
                Initialize the instance
              </Text>
              <Text c="dimmed" size="md">
                Before using the augmenter, we will first initialize an instance
                of the augmenter.
              </Text>
              <Text c="primary" size="md">
                Initialize Synonym Augmenter:
              </Text>
              <CodeHighlightTabs
                code={[
                  {
                    fileName: "python",
                    code: "syn_aug = SynonymAugmenter()",
                    language: "python",
                  },
                ]}
                className="w-full rounded-sm outline-1 outline outline-slate-600"
              />

              <Text c="primary" size="lg" fw="bolder">
                Calling the function
              </Text>
              <Text c="dimmed" size="md">
                After initializing the instance, you may call the augmenter
                function easily. For more information about the arguments
                required for each augmenter function, you may view them in the
                usage guide of other augmentation types:
              </Text>
              <Link
                href="/documentation/examples/random"
                className="flex ml-6 hover:text-primary"
              >
                -
                <Text className="ml-2 underline" ff="monospace" size="sm">
                  Random Augmenter
                </Text>
              </Link>
              <Link
                href="/documentation/examples/synonym"
                className="flex ml-6 hover:text-primary"
              >
                -
                <Text className="ml-2 underline" ff="monospace" size="sm">
                  Synonym Augmenter
                </Text>
              </Link>
              <Link
                href="/documentation/examples/llm"
                className="flex ml-6 hover:text-primary"
              >
                -
                <Text className="ml-2 underline" ff="monospace" size="sm">
                  Large Language Model (LLM) Augmenter
                </Text>
              </Link>
              <Text c="primary" size="md">
                Using Synonym Augmenter on a sentence:
              </Text>
              <CodeHighlightTabs
                code={[
                  {
                    fileName: "python",
                    code: "sample_text = \"The quick brown fox jumps over the lazy dog.\" \nsyn_aug(sample_text, aug_src='wordnet', aug_p=0.3)",
                    language: "python",
                  },
                  {
                    fileName: "output",
                    code: "'The speedy brown fox jumps over the slothful dog.'",
                    language: "bash",
                  },
                ]}
                className="w-full rounded-sm outline-1 outline outline-slate-600"
              />
              <Divider my={2} />
              <Text c="primary" size="lg" fw="bolder">
                Full code for the Synonym Augmenter Usage:
              </Text>
              <CodeHighlightTabs
                code={[
                  {
                    fileName: "python",
                    code: "from NLarge.synonym import SynonymAugmenter\n\nsyn_aug = SynonymAugmenter()\nsample_text = \"The quick brown fox jumps over the lazy dog.\"\nsyn_aug(sample_text, aug_src='wordnet', aug_p=0.3)",
                    language: "python",
                  },
                  {
                    fileName: "output",
                    code: "'The quick brown dodger jumps over the indolent dog.'",
                    language: "bash",
                  },
                ]}
                className="w-full rounded-sm outline-1 outline outline-slate-600"
              />
              <Text c="dimmed" size="md">
                Take note that the augmenter works on string level, this is done
                intentionally to provide for scalability and customization
                options. If you wish to perform augmenter on a dataset level,
                you would need to create helper functions to do so. You may find
                an example helper function to augment and enlarge the dataset in
                the helper functions guide in
                <Link
                  href="/documentation/helper"
                  className="underline ml-1 text-white hover:text-primary"
                >
                  Augmenting Datasets
                </Link>
                .
              </Text>

              <Divider my={2} />
            </Stack>
          </Group>
        </Group>
      </AppShellSection>
    </>
  );
}
