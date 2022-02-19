import org.junit.Test;

import java.util.Arrays;

public class AQ100Test {
    @Test
    public void bubbleSortTest() {
        AQ100 aq = new AQ100();
        int[] data = new int[]{3, 1, 2, 5};
        System.out.println(Arrays.toString(aq.t81_bubbleSort(data)));
    }
}
