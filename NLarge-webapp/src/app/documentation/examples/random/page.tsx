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
  { label: "Importing & Initialization", link: "#import", order: 2 },
  { label: "Random Swap", link: "#randomswap", order: 1 },
  { label: "Random Substitute", link: "#randomsubstitute", order: 1 },
  { label: "Random Delete", link: "#randomdelete", order: 1 },
  { label: "Random Crop", link: "#randomcrop", order: 1 },
  { label: "Code Example ", link: "#exampleRNN", order: 1 },
  { label: "Library Imports", link: "#exampleRNN_import", order: 2 },
  { label: "Dataset Download", link: "#exampleRNN_download", order: 2 },
  { label: "Dataset Augmentation", link: "#exampleRNN_augment", order: 2 },
  { label: "RNN: Model training", link: "#exampleRNN_training", order: 2 },
  { label: "RNN: Evaluating performance", link: "#exampleRNN_eval", order: 3 },
  { label: "LSTM: Model training", link: "#exampleLSTM_training", order: 2 },
  {
    label: "LSTM: Evaluating performance",
    link: "#exampleLSTM_eval",
    order: 3,
  },
  { label: "Analysis of Results", link: "#example_analysis", order: 2 },
];

export default function ExampleRandom() {
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
              Random Augmenter
            </Title>
            <Text c="dimmed" size="lg">
              Detailed guide to using the Random Augmenter. This page also
              serves as a proof of concept for the Random Augmenter.
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
              We will be explaining the different modes of the Random Augmenter,
              including an example using the &apos;Rotten Tomatoes&apos; dataset
              later on.
            </Text>
            <Text c="dimmed" size="md">
              As we can observe from the name of the augmenter, the Random
              Augmenter revolves around using a probability defined to modify a
              sequence. The random augmentation process involves iterating over
              each word in the sequence and performing the defined Action with a
              predefined probability. {"\n\n"}This introduces variability into
              the dataset, potentially improving the robustness and
              generalization capabilities of NLP models. Before we begin the
              explanation for each Action mode, let&apos;s first import and
              initialize the augmenter.
            </Text>
            <Text c="primary" size="lg" fw="bolder">
              Importing & initializing the library
            </Text>
            <div
              id="import"
              ref={(el) => {
                sectionRefs.current[1] = el;
              }}
            />
            <Text c="dimmed" size="md">
              Before using the library, you should first import and initialize
              the random augmenter. Since there are different modes to the
              Random Augmenter, be sure to import the Action class too!
            </Text>
            <Text c="primary" size="md">
              Import & Initialize NLarge Random Augmenter:
            </Text>
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: "from NLarge.random import RandomAugmenter, Action\nrandom_aug = RandomAugmenter()",
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />
            <Text c="dimmed" size="md">
              Great! Now let us go through each Random Augment Mode.
            </Text>
            <Divider my={2} />
            <Text c="primary" size="xl" fw="bolder">
              Random Swap
            </Text>
            <div
              id="randomswap"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[2] = el;
                }
              }}
            />
            <Text c="dimmed" size="md">
              The Swap Action randomly samples the target sequence with the
              predefined probability and swaps it&apos;s position with the
              adjacent words if the sampled word is not in the
              &apos;stop_words&apos; argument.
            </Text>
            <Text c="primary" size="lg" fw="bolder">
              Arguments:
            </Text>
            <Group>
              <Code bg="dimmed">data</Code>
              <Text>Input text to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">action</Code>
              <Text>
                Action to perform, in the case of using Random Swap,
                action=Action.SWAP
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_percent</Code>
              <Text>Percentage of words in sequence to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_min</Code>
              <Text>Minimum number of words to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_max</Code>
              <Text>Maximum number of words to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">skipwords</Code>
              <Text>List of words to skip augmentation</Text>
            </Group>
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: "input = \"This is a simple example sentence for testing.\"\nrandom_aug(data=input, action=Action.SWAP, aug_percent=0.3, aug_min=1, aug_max=10, skipwords=['is','a','for'])",
                  language: "python",
                },
                {
                  fileName: "output",
                  code: "('This is a simple example sentence for testing.', 'This is a sentence simple example for testing.')",
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />

            <Divider my={2} />

            <Text c="primary" size="xl" fw="bolder">
              Random Substitute
            </Text>
            <div
              id="randomsubstitute"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[3] = el;
                }
              }}
            />
            <Text c="dimmed" size="md">
              The Substitute Action randomly samples the target sequence with
              the predefined probability. It then substitutes the sampled
              word(s) with words chosen randomly in the provided
              &apos;target_words&apos; argument if the sampled word(s) is not in
              the &apos;stop_words&apos; argument.
            </Text>
            <Text c="primary" size="lg" fw="bolder">
              Arguments:
            </Text>
            <Group>
              <Code bg="dimmed">data</Code>
              <Text>Input text to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">action</Code>
              <Text>
                Action to perform, in the case of using Random Substitute,
                action=Action.SUBSTITUTE
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_percent</Code>
              <Text>Percentage of words in sequence to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_min</Code>
              <Text>Minimum number of words to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_max</Code>
              <Text>Maximum number of words to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">skipwords</Code>
              <Text>List of words to skip augmentation</Text>
            </Group>
            <Group>
              <Code bg="dimmed">target_words</Code>
              <Text>
                List of words to substitue with the original sampled word
              </Text>
            </Group>
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: "input = \"This is a simple example sentence for testing.\"\nrandom_aug(data=input, action=Action.SUBSTITUTE, aug_percent=0.3, aug_min=1, aug_max=10, skipwords=['is','a', 'for'], target_words=['great', 'awesome'])",
                  language: "python",
                },
                {
                  fileName: "output",
                  code: "('This is a simple example sentence for testing.', 'This is a simple great sentence for awesome')",
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />

            <Divider my={2} />

            <Text c="primary" size="xl" fw="bolder">
              Random Delete
            </Text>
            <div
              id="randomdelete"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[4] = el;
                }
              }}
            />
            <Text c="dimmed" size="md">
              The Delete Action randomly samples the target sequence with the
              predefined probability. It then deletes the sampled word if the
              sampled word is not in the &apos;stop_words&apos; argument.
            </Text>
            <Text c="primary" size="lg" fw="bolder">
              Arguments:
            </Text>
            <Group>
              <Code bg="dimmed">data</Code>
              <Text>Input text to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">action</Code>
              <Text>
                Action to perform, in the case of using Random Delete,
                action=Action.DELETE
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_percent</Code>
              <Text>Percentage of words in sequence to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_min</Code>
              <Text>Minimum number of words to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_max</Code>
              <Text>Maximum number of words to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">skipwords</Code>
              <Text>List of words to skip augmentation</Text>
            </Group>
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: "input = \"This is a simple example sentence for testing.\"\nrandom_aug(data=input, action=Action.DELETE, aug_percent=0.3, aug_min=1, aug_max=10, skipwords=['is','a', 'for'] )",
                  language: "python",
                },
                {
                  fileName: "output",
                  code: "('This is a simple example sentence for testing.', 'This is a sentence for testing.')",
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />

            <Divider my={2} />

            <Text c="primary" size="xl" fw="bolder">
              Random Crop
            </Text>
            <div
              id="randomcrop"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[5] = el;
                }
              }}
            />
            <Text c="dimmed" size="md">
              The Crop Action randomly samples a starting index and a ending
              index in the target sequence. The set of continuous words from the
              sampled starting to the sampled ending index will then be checked
              for the existence of stopwords. If the set of continuous words
              does not contain &apos;stopwords&apos;, it will be deleted.
            </Text>
            <Text c="primary" size="lg" fw="bolder">
              Arguments:
            </Text>
            <Group>
              <Code bg="dimmed">data</Code>
              <Text>Input text to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">action</Code>
              <Text>
                Action to perform, in the case of using Random Delete,
                action=Action.DELETE
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_percent</Code>
              <Text>Percentage of words in sequence to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_min</Code>
              <Text>Minimum number of words to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">aug_max</Code>
              <Text>Maximum number of words to augment</Text>
            </Group>
            <Group>
              <Code bg="dimmed">skipwords</Code>
              <Text>List of words to skip augmentation</Text>
            </Group>
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: "input = \"This is a simple example sentence for testing.\"\nrandom_aug(data=input, action=Action.CROP, aug_percent=0.3, aug_min=1, aug_max=10, skipwords=['is','a', 'for'] )",
                  language: "python",
                },
                {
                  fileName: "output",
                  code: "('This is a simple example sentence for testing.', 'This is a sentence for testing.')",
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />

            <Divider my={2} />

            <Text c="primary" size="xl" fw="bolder">
              Example of Random Augmentation
            </Text>
            <div
              id="exampleRNN"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[6] = el;
                }
              }}
            />
            <Text c="dimmed" size="md">
              For your reference, below is a full example of the NLarge Random
              Argumentation on a dataset. This example will also function as a
              proof of concept for the NLarge Random Augmentation. This example
              will be evaluating augmented datasets on RNN and LSTM based on the
              loss and accuracy metrics. We have chosen the &apos;rotten
              tomatoes&apos; dataset due to it&apos;s small size that is prone
              to overfitting.
            </Text>
            <Text c="primary" size="lg" fw="bolder">
              Importing libraries:
            </Text>
            <div
              id="exampleRNN_import"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[7] = el;
                }
              }}
            />
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: "import datasets \nfrom datasets import Dataset, Features, Value, concatenate_datasets \nfrom NLarge.dataset_concat import augment_data, MODE \nfrom NLarge.pipeline import TextClassificationPipeline\nfrom NLarge.model.RNN import TextClassifierRNN, TextClassifierLSTM",
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />
            <Text c="primary" size="lg" fw="bolder">
              Downloading &apos;rotten-tomatoes&apos; dataset
            </Text>
            <div
              id="exampleRNN_download"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[8] = el;
                }
              }}
            />
            <Text c="dimmed" size="md">
              Here, we download the dataset and ensure that the features are in
              the correct format for our dataset augmentation later on.
            </Text>
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: `
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
                  `,
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />

            <Text c="primary" size="lg" fw="bolder">
              Applying augmentation and enlarging dataset
            </Text>
            <div
              id="exampleRNN_augment"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[9] = el;
                }
              }}
            />
            <Text c="dimmed" size="md">
              We will be performing a 10% Random Substitute Augmentation and a
              100% Random Substitute Augmentation on the dataset. This would
              increase the dataset size by 10% and 100% respectively.
            </Text>
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: `
# Augment and increase size by 10% and 100%
percentages = {
    MODE.RANDOM.SUBSTITUTE: 0.1,  # 10% of data for random augmentation
}
augmented_data_list_10 = augment_data(original_train_data, percentages)

percentages = {
    MODE.RANDOM.SUBSTITUTE: 1.0,  # 100% of data for random augmentation
}
augmented_data_list_100 = augment_data(original_train_data, percentages)


# Convert augmented data into Datasets
augmented_dataset_10 = Dataset.from_dict(
    {
        "text": [item["text"] for item in augmented_data_list_10],
        "label": [item["label"] for item in augmented_data_list_10],
    },
    features=features,
)

augmented_dataset_100 = Dataset.from_dict(
    {
        "text": [item["text"] for item in augmented_data_list_100],
        "label": [item["label"] for item in augmented_data_list_100],
    },
    features=features,
)

# Concatenate original and augmented datasets
augmented_train_data_10 = concatenate_datasets(
    [original_train_data, augmented_dataset_10]
)

augmented_train_data_100 = concatenate_datasets(
    [original_train_data, augmented_dataset_100]
)
                  `,
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />

            <Text c="primary" size="lg" fw="bolder">
              RNN: Loading the pipeline & model training
            </Text>
            <div
              id="exampleRNN_training"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[10] = el;
                }
              }}
            />
            <Text c="dimmed" size="md">
              Here, we will initialize and train the pipeline using RNN and the
              augmented datasets.
            </Text>
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: `
pipeline_augmented_10 = TextClassificationPipeline(
    augmented_data=augmented_train_data_10,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=TextClassifierRNN,
)
pipeline_augmented_100 = TextClassificationPipeline(
    augmented_data=augmented_train_data_100,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=TextClassifierRNN,
)
pipeline_augmented_10.train_model(n_epochs=10)
pipeline_augmented_100.train_model(n_epochs=10)
                  `,
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />

            <Text c="primary" size="lg" fw="bolder">
              RNN: Evaluating the models&apos; performance
            </Text>
            <div
              id="exampleRNN_eval"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[11] = el;
                }
              }}
            />
            <Text c="dimmed" size="md">
              Plotting the loss and accuracy graphs, we can visualize the
              performance improvements between the two amount of augmentation on
              RNN.
            </Text>
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: `
pipeline_augmented_10.plot_loss(title="10% Random Substitute on RNN")
pipeline_augmented_100.plot_loss(title="100% Random Substitute on RNN")
pipeline_augmented_10.plot_acc(title="10% Random Substitute on RNN")
pipeline_augmented_100.plot_acc(title="100% Random Substitute on RNN")

                  `,
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />
            <Text c="dimmed" size="md">
              Looking at the graphs, we can see a stark improvement on both loss
              and accuracy of the RNN model.
            </Text>
            <Text c="primary" size="md">
              Models&apos; Loss
            </Text>
            <Image src="/graphs/rnn_random_sub_10_loss.png" />
            <Image src="/graphs/rnn_random_sub_100_loss.png" />
            <Text c="primary" size="md">
              Models&apos; Accuracy
            </Text>
            <Image src="/graphs/rnn_random_sub_10_acc.png" />
            <Image src="/graphs/rnn_random_sub_100_acc.png" />

            <Divider my={2} />

            <Text c="primary" size="lg" fw="bolder">
              LSTM: Loading the pipeline & model training
            </Text>
            <div
              id="exampleLSTM_training"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[12] = el;
                }
              }}
            />
            <Text c="dimmed" size="md">
              Here, we will initialize and train the pipeline using LSTM and the
              augmented datasets.
            </Text>
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: `
pipeline_augmented_10_LSTM = TextClassificationPipeline(
    augmented_data=augmented_train_data_10,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=TextClassifierLSTM,
)
pipeline_augmented_100_LSTM = TextClassificationPipeline(
    augmented_data=augmented_train_data_100,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=TextClassifierLSTM,
)
pipeline_augmented_10_LSTM.train_model(n_epochs=10)
pipeline_augmented_100_LSTM.train_model(n_epochs=10)
                  `,
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />

            <Text c="primary" size="lg" fw="bolder">
              LSTM: Evaluating the models&apos; performance
            </Text>
            <div
              id="exampleLSTM_eval"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[13] = el;
                }
              }}
            />
            <Text c="dimmed" size="md">
              Plotting the loss and accuracy graphs, we can visualize the
              performance improvements between the two amount of augmentation
              when used on LSTM.
            </Text>
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: `
pipeline_augmented_10_LSTM.plot_loss(title="10% Random Substitute on LSTM")
pipeline_augmented_100_LSTM.plot_loss(title="100% Random Substitute on LSTM")
pipeline_augmented_10_LSTM.plot_acc(title="10% Random Substitute on LSTM")
pipeline_augmented_100_LSTM.plot_acc(title="100% Random Substitute on LSTM")

                  `,
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />
            <Text c="dimmed" size="md">
              Looking at the graphs, we can see a stark improvement on both loss
              and accuracy of the LSTM model.
            </Text>
            <Text c="primary" size="md">
              Models&apos; Loss
            </Text>
            <Image src="/graphs/lstm_random_sub_10_loss.png" />
            <Image src="/graphs/lstm_random_sub_100_loss.png" />
            <Text c="primary" size="md">
              Models&apos; Accuracy
            </Text>
            <Image src="/graphs/lstm_random_sub_10_acc.png" />
            <Image src="/graphs/lstm_random_sub_100_acc.png" />

            <Text c="primary" size="lg" fw="bolder">
              Analysis of Results
            </Text>
            <div
              id="example_analysis"
              ref={(el) => {
                if (el) {
                  sectionRefs.current[14] = el;
                }
              }}
            />
            <Text c="dimmed" size="md">
              The results of our experiment indicate that the performance of the
              models keeps increasing with higher levels of augmentation. This
              suggests that data augmentation provides a clear benefit for
              sentiment classification tasks. Additionally, the findings
              highlight the importance of data augmentation in enhancing the
              diversity and robustness of training datasets, leading to imporved
              model performance.
            </Text>
            <Text c="dimmed" size="md">
              The data augmentation techniques mitigates overfitting by
              effectively increasing the size of the training dataset, reducing
              the likelihood of the model memorizing specific examples and
              encouraging it to learn general patterns instead. The introduction
              of variations in the training data makes the model more robust to
              noise and variations in real world input data, which is crucial
              for achieving good performance on unseen data.
            </Text>
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
