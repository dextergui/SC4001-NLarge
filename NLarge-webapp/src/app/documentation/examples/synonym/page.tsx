"use client";
import { CodeHighlightTabs } from "@mantine/code-highlight";
import {
  AppShellSection,
  Box,
  Code,
  Divider,
  Group,
  Image,
  rem,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import clsx from "clsx";
import classes from "@/components/TableOfContents.module.css";
import { useEffect, useRef, useState } from "react";
import { IconListSearch } from "@tabler/icons-react";

const links = [
  { label: "Introduction", link: "#introduction", order: 1 },
  { label: "Key Components", link: "#keyComponents", order: 1 },
  { label: "Importing SynonymAugmenter", link: "#import", order: 1 },
  { label: "Parameters", link: "#parameters", order: 1 },
  { label: "Single Sentence Usage", link: "#singleUsage", order: 1 },
  { label: "Full Example", link: "#fullExample", order: 1 },
  { label: "Full Code ", link: "#fullCode", order: 2 },
  { label: "Model Loss", link: "#modelLoss", order: 2 },
  { label: "Model Accuracy", link: "#modelAcc", order: 2 },
];

export default function ExampleSynonym() {
  const [TOCactive, setTOCactive] = useState(0);
  const sectionRefs = useRef<(HTMLDivElement | null)[]>([]); // Store references to each section

  // Set active TOC item based on scroll position
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            // Update active TOC item based on visible section
            const sectionId = entry.target.id;
            setTOCactive(
              links.findIndex((link) => link.link === `#${sectionId}`),
            );
          }
        });
      },
      { threshold: 0.6 }, // Trigger when 60% of the section is visible
    );

    // Observe each section
    sectionRefs.current.forEach((section) => {
      if (section) {
        observer.observe(section);
      }
    });

    return () => {
      observer.disconnect();
    };
  }, []);

  // Scroll to the respective section when TOC item is clicked
  const handleTOCClick = (link: string) => {
    const targetSection = document.querySelector(link);
    if (targetSection) {
      window.scrollTo({
        top: targetSection.getBoundingClientRect().top + window.scrollY - 300, // Adjust scroll position with -80px offset for the header
        behavior: "smooth",
      });
    }
  };

  const items = links.map((item, index) => (
    <Box<"a">
      component="a"
      href={item.link}
      onClick={(event) => {
        event.preventDefault();
        handleTOCClick(item.link);
      }}
      key={item.label}
      className={clsx(classes.link, {
        [classes.linkActive]: TOCactive === index,
      })}
      style={{ paddingLeft: `calc(${item.order} * var(--mantine-spacing-md))` }}
    >
      {item.label}
    </Box>
  ));
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
              Synonym Augmenter
            </Title>
            <Text c="dimmed" size="lg">
              Detailed guide to using the Synonym Augmenter. This page also
              serves as a proof of concept for the Synonym Augmenter.
            </Text>
          </Stack>
        </Group>
        <Divider my="lg" />
      </AppShellSection>
      <AppShellSection className="flex justify-center pb-10">
        <div className="max-w-screen-lg  w-full flex space-x-8 ml-16">
          <Stack className="w-2/3">
            <Text c="primary" size="lg" fw="bolder">
              Introduction
            </Text>
            <div
              id="introduction"
              ref={(el) => {
                sectionRefs.current[0] = el;
              }}
            />
            <Text c="dimmed" size="md">
              The Synonym Augmenter enables data augmentation for text sentiment
              classification by introducing variability in text through synonym
              replacement. This augmenter enhances a dataset by augmenting words
              with their synonyms, which can improve model robustness by
              introducing semantic variability without changing a sentiment.
              <br />
              <br /> The Synonym Augmenter samples word in the target sequence
              with a predefined probability and replace it with a randomly
              chosen synonym from a set of synonyms of the sampled word.
              <br /> <br />
              <i>
                In the current version of NLarge, the set of synonyms can also
                be drawn from <Code bg="dimmed">WordNet</Code>, an extensive
                lexical database.
              </i>
            </Text>
            <Text c="primary" size="lg" fw="bolder">
              Key Components
            </Text>
            <div
              id="keyComponents"
              ref={(el) => {
                sectionRefs.current[1] = el;
              }}
            />
            <Text c="primary" size="md" fw="bolder">
              WordNet
            </Text>
            <Text c="dimmed" size="md">
              <Code bg="dimmed">WordNet</Code> provides synonym and antonym
              lookup, with optional parts of speech (POS) filtering. The POS
              tagging functionality identifies relevant grammatical structures
              for more accurate augmentation.
            </Text>
            <Text c="primary" size="md" fw="bolder">
              PartsOfSpeech
            </Text>
            <Text c="dimmed" size="md">
              Our POS functionality maps between POS tags and constituent tags
              to ensure compatibility with <Code bg="dimmed">WordNet</Code>
              &apos;s POS requirements. <br /> <br />{" "}
              <i>
                The current version supports noun, verb, adjective and adverb
                classifications.
              </i>
            </Text>
            <Text c="primary" size="md" fw="bolder">
              SynonymAugmenter
            </Text>
            <Text c="dimmed" size="md">
              The augmenter uses the <Code bg="dimmed">WordNet</Code> class to
              perform augmentation by replacing words with synonyms based on
              user-defined criteria. It utilizes POS tagging to determine
              eligible words for substituition, while skip lists (stop words and
              regex patterns) can prevent certain words from being replaced.
            </Text>

            <Text c="primary" size="lg" fw="bolder">
              Import & Initialize NLarge Synonym Augmenter
            </Text>
            <div
              id="import"
              ref={(el) => {
                sectionRefs.current[2] = el;
              }}
            />
            <Text c="dimmed" size="md">
              Before we proceed further, let us first import and initialize the
              SynonymAugmenter instance.
            </Text>
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: `
from NLarge.synonym import SynonymAugmenter

syn_aug = SynonymAugmenter()
`,
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />

            <Text c="primary" size="xl" fw="bolder">
              Parameters
            </Text>
            <div
              id="parameters"
              ref={(el) => {
                sectionRefs.current[3] = el;
              }}
            />
            <Group>
              <Code bg="dimmed">data</Code>
              <Text>
                (str) - Input text to augment
                <br /> <i>example: &apos;This is a test sentence.&apos;</i>
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_src</Code>
              <Text>
                (str) - Augmentation source, currently supports only
                &quot;wordnet&quot;. <br />
                <i>default: &apos;wordnet&apos;</i>
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">lang</Code>
              <Text>
                (str) - Language of the input text. <br />{" "}
                <i>default: &apos;eng&apos;</i>
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_max</Code>
              <Text>
                (int) - Maximum number of words to augment.
                <br /> <i>default: 10</i>
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_p</Code>
              <Text>
                (float) - Probability of augmenting each word. <br />{" "}
                <i>default: 0.3</i>
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">stopwords</Code>
              <Text>
                (list) - List of words to exclude from augmentation. <br />{" "}
                <i>default: None</i>
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">tokenizer</Code>
              <Text>
                (function) - Function to tokenize the input text. <br />{" "}
                <i>default: None</i>
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">reverse_tokenizer</Code>
              <Text>
                (function) - Function to detokenize the augmented text. <br />{" "}
                <i>default: None</i>
              </Text>
            </Group>
            <Text c="primary" size="xl" fw="bolder">
              Single Sentence Usage Example
            </Text>
            <div
              id="singleUsage"
              ref={(el) => {
                sectionRefs.current[4] = el;
              }}
            />
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: `
sample_text = "The quick brown fox jumps over the lazy dog."
print(sample_text)
syn_aug(sample_text, aug_src='wordnet', aug_p=0.3, aug_max=20)
                  `,
                  language: "python",
                },
                {
                  fileName: "output",
                  code: `
The quick brown fox jumps over the lazy dog.
'The quick brown fox leap over the faineant dog.'
                  `,
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />

            <Divider my={2} />

            <Text c="primary" size="xl" fw="bolder">
              Full Example of Synonym Augmentation
            </Text>
            <div
              id="fullExample"
              ref={(el) => {
                sectionRefs.current[5] = el;
              }}
            />
            <Text c="dimmed" size="md">
              For your reference, below is a full example of the NLarge Synonym
              Argumentation on a dataset. This example will also function as a
              proof of concept for the NLarge Synonym Augmentation. This example
              will be evaluating augmented datasets on LSTM based on the loss
              and accuracy metrics. We have chosen the &apos;rotten
              tomatoes&apos; dataset due to it&apos;s small size that is prone
              to overfitting.
            </Text>
            <Text c="primary" size="lg" fw="bolder">
              Full Code:
            </Text>
            <div
              id="fullCode"
              ref={(el) => {
                sectionRefs.current[6] = el;
              }}
            />
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: `
import datasets
from datasets import Dataset, Features, Value, concatenate_datasets
from NLarge.dataset_concat import augment_data, MODE
from NLarge.pipeline import TextClassificationPipeline
from NLarge.model.RNN import TextClassifierLSTM                  

original_train_data, original_test_data = datasets.load_dataset(
"rotten_tomatoes", split=["train", "test"]
)  
features = Features({"text": Value("string"), "label": Value("int64")})
original_train_data = Dataset.from_dict(
    {
        "text": original_train_data["text"],
        "label": original_train_data["label"],
    },
    features=features,
)  

# Augment and increase size by 5%, 10%
percentage= {
    MODE.SYNONYM.WORDNET: 0.05,
}
augmented_synonym_5 = augment_data(original_train_data, percentage)

percentage= {
    MODE.SYNONYM.WORDNET: 0.10,
}
augmented_synonym_10 = augment_data(original_train_data, percentage)

# Convert augmented data into Datasets
augmented_dataset_5 = Dataset.from_dict(
    {
        "text": [item["text"] for item in augmented_synonym_5],
        "label": [item["label"] for item in augmented_synonym_5],
    },
    features=features,
)
augmented_dataset_10 = Dataset.from_dict(
    {
        "text": [item["text"] for item in augmented_synonym_10],
        "label": [item["label"] for item in augmented_synonym_10],
    },
    features=features,
)

# Concatenate original and augmented datasets
augmented_train_data_5 = concatenate_datasets(
    [original_train_data, augmented_dataset_5]
)
augmented_train_data_10 = concatenate_datasets(
    [original_train_data, augmented_dataset_10]
)

# Initialize Pipelines
pipeline_augmented_5 = TextClassificationPipeline(
    augmented_data=augmented_train_data_5,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=TextClassifierLSTM,
)
pipeline_augmented_10 = TextClassificationPipeline(
    augmented_data=augmented_train_data_10,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=TextClassifierLSTM,
)

# Train Models
pipeline_augmented_5.train_model(n_epochs=10)
pipeline_augmented_10.train_model(n_epochs=10)

# Plot Loss 
pipeline_augmented_5.plot_loss(title="5% Synonym Augment on LSTM")
pipeline_augmented_10.plot_loss(title="10% Synonym Augment on LSTM")

# Plot Accuracy
pipeline_augmented_5.plot_acc(title="5% Synonym Augment on LSTM")
pipeline_augmented_10.plot_acc(title="10% Synonym Augment on LSTM")

                  `,
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />
            <Text c="primary" size="md">
              Models&apos; Loss
            </Text>
            <div
              id="modelLoss"
              ref={(el) => {
                sectionRefs.current[7] = el;
              }}
            />
            <Group justify="center">
              <Image src="/graphs/lstm_synonym_5_loss.png" />
              <Image src="/graphs/lstm_synonym_10_loss.png" />
            </Group>

            <Text c="primary" size="md">
              Models&apos; Accuracy
            </Text>
            <div
              id="modelAcc"
              ref={(el) => {
                sectionRefs.current[8] = el;
              }}
            />
            <Group justify="center">
              <Image src="/graphs/lstm_synonym_5_acc.png" />
              <Image src="/graphs/lstm_synonym_10_acc.png" />
            </Group>

            <Divider my={2} />
          </Stack>

          <div className={clsx(classes.root, "w-1/4 sticky top-8")}>
            <Group mb="md">
              <IconListSearch
                style={{ width: rem(18), height: rem(18) }}
                stroke={1.5}
              />
              <Text>Table of contents</Text>
            </Group>
            <div className={classes.links}>
              <div
                className={classes.indicator}
                style={{
                  transform: `translateY(calc(${TOCactive} * var(--link-height) + var(--indicator-offset)))`,
                }}
              />
              {items}
            </div>
          </div>
        </div>
      </AppShellSection>
    </>
  );
}
