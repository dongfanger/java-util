import org.junit.Test;

import java.util.Arrays;

public class AQ80Test {
    @Test
    public void bubbleSortTest() {
        AQ80 aq = new AQ80();
        int[] data = new int[]{3, 1, 2, 5};
        System.out.println(Arrays.toString(aq.bubbleSort(data)));
    }
}
