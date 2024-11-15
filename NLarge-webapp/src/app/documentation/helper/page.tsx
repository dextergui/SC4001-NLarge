import { CodeHighlightTabs } from "@mantine/code-highlight";
import {
  AppShellSection,
  Code,
  Divider,
  Group,
  List,
  ListItem,
  rem,
  Stack,
  Text,
  Title,
} from "@mantine/core";

export default function DocumentationHelper() {
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
              Helper Functions/Classes
            </Title>
            <Text c="dimmed" size="lg">
              Here are some helper functions and classes we designed to aid the
              usage of NLarge in a DNN usage.
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
                dataset_concat
              </Text>
              <Text c="dimmed" size="md">
                You may import the <Code bg="dimmed">augment_data</Code>{" "}
                function and the <Code bg="dimmed">MODE</Code> class from this
                library file.
              </Text>
              <Text c="primary" size="lg" fw="bolder">
                MODE class
              </Text>
              <Text c="dimmed" size="md">
                The <Code bg="dimmed">MODE</Code> class categorizes the
                available augmentation techniques in NLarge. It is used with the{" "}
                <Code bg="dimmed">augment_data</Code> function.
              </Text>
              <Text c="primary" size="md" fw="bolder">
                Augmentation Modes
              </Text>
              <List>
                <ListItem>
                  <Code bg="dimmed" fz="sm">
                    RANDOM
                  </Code>
                  <List withPadding listStyleType="disc" className="pt-2">
                    <ListItem>
                      <Text c="dimmed" size="md">
                        <Code bg="dimmed" fz="sm">
                          SWAP
                        </Code>{" "}
                        : Swaps words within the text.
                      </Text>
                    </ListItem>
                    <ListItem>
                      <Text c="dimmed" size="md">
                        <Code bg="dimmed" fz="sm">
                          SUBSTITUTE
                        </Code>{" "}
                        : Substitutes words with other words.
                      </Text>
                    </ListItem>
                    <ListItem>
                      <Text c="dimmed" size="md">
                        <Code bg="dimmed" fz="sm">
                          DELETE
                        </Code>{" "}
                        : Deletes certain words from the text.
                      </Text>
                    </ListItem>
                    <ListItem>
                      <Text c="dimmed" size="md">
                        <Code bg="dimmed" fz="sm">
                          CROP
                        </Code>{" "}
                        : Trims the text.
                      </Text>
                    </ListItem>
                  </List>
                </ListItem>
                <ListItem className="pt-2">
                  <Code bg="dimmed" fz="sm">
                    SYNONYM
                  </Code>
                  <List withPadding listStyleType="disc" className="pt-2">
                    <ListItem>
                      <Text c="dimmed" size="md">
                        <Code bg="dimmed" fz="sm">
                          WORDNET
                        </Code>{" "}
                        : Replaces words with synonyms from WordNet.
                      </Text>
                    </ListItem>
                  </List>
                </ListItem>
                <ListItem className="pt-2">
                  <Code bg="dimmed" fz="sm">
                    LLM
                  </Code>
                  <List withPadding listStyleType="disc" className="pt-2">
                    <ListItem>
                      <Text c="dimmed" size="md">
                        <Code bg="dimmed" fz="sm">
                          PARAPHRASE
                        </Code>{" "}
                        : Paraphrases the text using a language model.
                      </Text>
                    </ListItem>
                    <ListItem>
                      <Text c="dimmed" size="md">
                        <Code bg="dimmed" fz="sm">
                          SUMMARIZE
                        </Code>{" "}
                        : Summarize the text using a language model.
                      </Text>
                    </ListItem>
                  </List>
                </ListItem>
              </List>

              <Text c="primary" size="lg" fw="bolder">
                augment_data Function
              </Text>
              <Text c="dimmed" size="md">
                The <Code bg="dimmed">augment_data</Code> function enables the
                generation of new samples from an existing dataset using the
                augmentation techniques provided in NLarge, including random
                transformations, synonym replacement, and language model-based
                paraphrasing or summarization. This function allows users to
                specify percentages for different augmentation modes and stack
                multiple augmentation modes to diversify and enlarge the
                dataset, which can help improve model robustness and prevent
                overfitting.
              </Text>
              <Text c="primary" size="md" fw="bolder">
                Parameters
              </Text>
              <Group>
                <Text>
                  <Code bg="dimmed">dataset</Code> <i>(Dataset)</i>
                </Text>
                <Text c="dimmed">
                  The original dataset to augment, structured with at least two
                  fields: &quot;text&quot; for the input text and
                  &quot;label&quot; for associated labels.
                </Text>
              </Group>
              <Group>
                <Text>
                  <Code bg="dimmed">percentages</Code> <i>(dict)</i>
                </Text>
                <Text c="dimmed">
                  A dictionary specifying the augmentation techniques to apply
                  and the percentages of data to be augmented by each technique.
                  Keys are augmentation modes (from the{" "}
                  <Code bg="dimmed">MODE</Code> class) and values ate float
                  numbers representing the percentage of samples for each
                  augmentation.
                </Text>
              </Group>
              <Text c="primary" size="md" fw="bolder">
                Returns
              </Text>
              <Text c="dimmed" size="md">
                The function returns a list of augmented dataset samples. Each
                sample is a dictionary with &quot;text&quot; (augmented text)
                and &quot;label&quot; (original label) fields.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                Example Usage:
              </Text>
              <CodeHighlightTabs
                code={[
                  {
                    fileName: "python",
                    code: `
from NLarge.dataset_concat import augment_data, MODE

# Augment and increase size by 100%
percentages = {
    MODE.RANDOM.SUBSTITUTE: 0.5,  # 50% of data for random augmentation
    MODE.SYNONYM.WORDNET: 0.5,  # 50% of data for synonym augmentation
}

augmented_data_list = augment_data(original_train_data, percentages)
                    `,
                    language: "python",
                  },
                ]}
                className="w-full rounded-sm outline-1 outline outline-slate-600"
              />
              <Divider my={2} />

              <Text c="primary" size="lg" fw="bolder">
                pipeline
              </Text>
              <Text c="dimmed" size="md">
                You may import the{" "}
                <Code bg="dimmed">TextClassifierPipeline</Code> class from this
                library file.
              </Text>
              <Text c="primary" size="md" fw="bolder">
                TextClassificationPipeline
              </Text>
              <Text c="dimmed" size="md">
                The <Code bg="dimmed">TextClassifierPipeline</Code> class in the
                NLarge library is designed to streamline the process of setting
                up and training a text classification model. It handles data
                preprocessing, vocabulary creation, model instantiation, and
                evaluation, enabling users to initialize a complete pipeline
                with minimal setup.
              </Text>
              <Text c="primary" size="md" fw="bold">
                Key Features
              </Text>
              <List
                type="ordered"
                listStyleType="number"
                size="lg"
                c="dimmed"
                spacing="sm"
              >
                <ListItem fw="bold">
                  Data Preparation
                  <List withPadding listStyleType="disc" fw="normal">
                    <ListItem>
                      Tokenizes text, builds vocabulary, and numericalizes text
                      data.
                    </ListItem>
                    <ListItem>
                      Supports handling both augmented and test datasets, and
                      splits the augmented dataset into training and validation
                      sets.
                    </ListItem>
                    <ListItem>
                      Automatically configures PyTorch DataLoaders to handle
                      batching and padding.
                    </ListItem>
                  </List>
                </ListItem>
                <ListItem fw="bold">
                  Model Initialization
                  <List withPadding listStyleType="disc" fw="normal">
                    <ListItem>
                      Instantiates a text classification model based on the
                      <Code bg="dimmed">TextClassifierRNN</Code> class.
                    </ListItem>
                    <ListItem>
                      Loads pre-trained embeddings (GloVe) for initializing word
                      vectors, improving performance for text representations.
                    </ListItem>
                    <ListItem>
                      Allows users to specify key hyperparameters such as
                      embedding dimension, hidden dimension, number of layers,
                      dropout rate, and learning rate.
                    </ListItem>
                  </List>
                </ListItem>
                <ListItem fw="bold">
                  Training and Evaluation:
                  <List withPadding listStyleType="disc" fw="normal">
                    <ListItem>
                      Includes methods for training (
                      <Code bg="dimmed">train_model</Code>) and evaluation (
                      <Code bg="dimmed">evaluate</Code>) of the model, using
                      accuracy and cross-entropy loss.
                    </ListItem>
                    <ListItem>
                      Tracks training and validation loss and accuracy over
                      epochs.
                    </ListItem>
                    <ListItem>
                      Supports saving the best model weights during training
                      based on validation performance.
                    </ListItem>
                  </List>
                </ListItem>
                <ListItem fw="bold">
                  Visualization:
                  <List withPadding listStyleType="disc" fw="normal">
                    <ListItem>
                      Provides <Code bg="dimmed">plot_loss</Code> and{" "}
                      <Code bg="dimmed">plot_acc</Code> methods to visualize the
                      training and validation loss and accuracy over epochs,
                      aiding in monitoring model convergence and generalization.
                    </ListItem>
                  </List>
                </ListItem>
              </List>
              <Text c="primary" size="md" fw="bold">
                Pipeline Initialization
              </Text>
              <Text c="dimmed" size="md">
                To initialize a pipline, users need to provide:
              </Text>
              <List withPadding spacing="md" listStyleType="disc" c="dimmed">
                <ListItem>
                  augmented_data: The training dataset, ideally augmented with
                  techniques provided by NLarge to imporve robustness
                </ListItem>
                <ListItem>
                  test_data: The test dataset for final evaluation
                </ListItem>
                <ListItem>
                  hyperparameters: Key parameters like{" "}
                  <Code bg="dimmed">batch_size</Code>,{" "}
                  <Code bg="dimmed">embedding_dim</Code>,{" "}
                  <Code bg="dimmed">hidden_dim</Code>,{" "}
                  <Code bg="dimmed">n_layers</Code>,{" "}
                  <Code bg="dimmed">dropout_rate</Code> and{" "}
                  <Code bg="dimmed">lr</Code>
                </ListItem>
              </List>
              <Text c="primary" size="md" fw="bold">
                Example Initialization
              </Text>
              <CodeHighlightTabs
                code={[
                  {
                    fileName: "python",
                    code: `
# Import Libraries
from NLarge.pipeline import TextClassificationPipeline
from NLarge.model.RNN import TextClassifierRNN

# Initialize Pipeline
pipeline_augmented = TextClassificationPipeline(
    augmented_data=augmented_train_data,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=TextClassifierRNN,
)

# Train Pipeline
pipeline_augmented.train_model(n_epochs=10)
                    `,
                    language: "python",
                  },
                ]}
                className="w-full rounded-sm outline-1 outline outline-slate-600"
              />
            </Stack>
          </Group>
        </Group>
      </AppShellSection>
    </>
  );
}
